"""
Proper steering operator (Innovation 1) + operator-works test.
Monkey-patches Qwen3-VL eager attention to add a bias to IMAGE-KEY attention logits for a
TARGET head-set (the localized provenance heads), at re-look decode steps. Tests whether, in
the SELF condition (illusion), the operator raises visual-attention mass S_vis toward the
USER-condition level. This is the mechanism-level proof that the fix moves the right quantity.
"""
import os, sys, json, argparse
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, numpy as np, pandas as pd
from PIL import Image
import io
from transformers import AutoProcessor
import transformers.models.qwen3_vl.modeling_qwen3_vl as qvl
from qwen_vl_utils import process_vision_info

CUE=" Wait, let me check the figure again to make sure I haven't made a mistake."
STEER={"on":False,"img_mask":None,"alpha":6.0,"target":None}  # target: dict layer_idx->set(head)

# ---- proper attention patch: add alpha to image-key logits for target heads ----
_orig=qvl.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states=qvl.repeat_kv(key, module.num_key_value_groups)
    value_states=qvl.repeat_kv(value, module.num_key_value_groups)
    attn=torch.matmul(query, key_states.transpose(2,3))*scaling
    if attention_mask is not None:
        attn=attn+attention_mask[:,:,:,:key_states.shape[-2]]
    if STEER["on"] and STEER["img_mask"] is not None and STEER["target"] is not None:
        li=getattr(module,"layer_idx",None)
        heads=STEER["target"].get(li)
        if heads:
            k=attn.shape[-1]; im=STEER["img_mask"]
            if im.shape[0]>=k: colmask=im[:k]
            else:
                colmask=torch.zeros(k,dtype=torch.bool,device=attn.device); colmask[:im.shape[0]]=im
            add=torch.zeros(attn.shape[1],device=attn.device,dtype=attn.dtype)
            for h in heads:
                if h<add.shape[0]: add[h]=STEER["alpha"]
            attn=attn+add[None,:,None,None]*colmask[None,None,None,:].to(attn.dtype)
    attn=torch.nn.functional.softmax(attn,dim=-1,dtype=torch.float32).to(query.dtype)
    attn=torch.nn.functional.dropout(attn,p=dropout,training=module.training)
    out=torch.matmul(attn,value_states); out=out.transpose(1,2).contiguous()
    return out, attn
qvl.eager_attention_forward=patched

def load(snap, dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    m=Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager",
                          device_map={"":dev}, local_files_only=True).eval()
    return proc, m

def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")

@torch.no_grad()
def reason(proc, model, dev, img, q, mx=140):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+"\nThink briefly, then answer."}]}]
    t=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im,vi=process_vision_info(msgs); inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

@torch.no_grad()
def svis_self(proc, model, dev, img, q, R, n_steps=24, steer=False):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+"\nThink briefly, then answer."}]},
          {"role":"assistant","content":[{"type":"text","text":R+CUE}]}]
    t=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    im,vi=process_vision_info(msgs[:1]); inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    STEER["img_mask"]=(inp["input_ids"][0]==model.config.image_token_id)
    STEER["on"]=steer
    out=model.generate(**inp,max_new_tokens=n_steps,do_sample=False,output_attentions=True,return_dict_in_generate=True)
    STEER["on"]=False
    L=len(out.attentions[0]); im_mask=STEER["img_mask"]; acc=0.0; cnt=0
    for step in out.attentions:
        for li in range(L):
            a=step[li][0,:,-1,:]
            acc+=float(a[:,:im_mask.shape[0]][:,im_mask].sum(-1).mean()); cnt+=1
    return acc/max(cnt,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True)
    ap.add_argument("--prov_diff",required=True, help="prov_diff_C2C1.npy [L,H]")
    ap.add_argument("--topk",type=int,default=60); ap.add_argument("--alpha",type=float,default=6.0)
    ap.add_argument("--n",type=int,default=30); ap.add_argument("--dev",default="cuda:0")
    a=ap.parse_args(); STEER["alpha"]=a.alpha
    pd_=np.load(a.prov_diff)                      # [L,H] provenance differential
    L,H=pd_.shape; flat=pd_.flatten(); top=np.argsort(flat)[::-1][:a.topk]
    target={}
    for idx in top: target.setdefault(int(idx//H),set()).add(int(idx%H))
    STEER["target"]=target
    print(f"steering {a.topk} heads across {len(target)} layers, alpha={a.alpha}", flush=True)
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet)
    rows=[]
    for i in range(min(a.n,len(df))):
        r=df.iloc[i]
        try: img=pil(r["image"]); q=str(r["question"])
        except: continue
        R=reason(proc,model,a.dev,img,q)
        off=svis_self(proc,model,a.dev,img,q,R,steer=False)
        on =svis_self(proc,model,a.dev,img,q,R,steer=True)
        rows.append({"i":i,"self_off":off,"self_steer":on,"delta":on-off})
        if (i+1)%5==0:
            d=pd.DataFrame(rows)
            print(f"[{i+1}] self_off={d.self_off.mean():.4f} self_steer={d.self_steer.mean():.4f} delta={d.delta.mean():+.4f}",flush=True)
    d=pd.DataFrame(rows)
    def boot(x):
        x=np.asarray(x); idx=np.random.RandomState(0).randint(0,len(x),(2000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== OPERATOR TEST N={len(d)} topk={a.topk} alpha={a.alpha} ===")
    for c in ("self_off","self_steer","delta"):
        m,lo,hi=boot(d[c].values); print(f"  {c:11s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("  (recall C1 self~0.019, C2 user~0.029; want self_steer to rise toward/past C2)")

if __name__=="__main__": main()
