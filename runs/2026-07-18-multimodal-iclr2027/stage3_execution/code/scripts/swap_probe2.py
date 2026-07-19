"""
Behavioral illusion + PROPER-operator fix (real VisualSwap images).
Same as swap_probe.py but self+STEER uses the proper per-head image-key ATTENTION-BIAS operator
(monkey-patch on the localized provenance heads), not crude hidden-state scaling.
Metric: logit(Changed) > logit(Same) after a mid-reasoning image swap. self=illusion, user=recovery cue.
"""
import os, json, argparse
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, numpy as np, pandas as pd
from PIL import Image
import io
from transformers import AutoProcessor
import transformers.models.qwen3_vl.modeling_qwen3_vl as qvl
from qwen_vl_utils import process_vision_info

CUE="Wait, let me check the figure again to make sure I haven't made a mistake."
CHECK=" Look carefully at the image shown right now. Is it the SAME figure described in the analysis above, or has it changed? Answer with one word: Same or Changed."
PRIME="\nMy one-word answer (Same or Changed) is:"
STEER={"on":False,"img_mask":None,"alpha":6.0,"target":None}

def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks=qvl.repeat_kv(key, module.num_key_value_groups); vs=qvl.repeat_kv(value, module.num_key_value_groups)
    attn=torch.matmul(query, ks.transpose(2,3))*scaling
    if attention_mask is not None: attn=attn+attention_mask[:,:,:,:ks.shape[-2]]
    if STEER["on"] and STEER["img_mask"] is not None and STEER["target"] is not None:
        li=getattr(module,"layer_idx",None); heads=STEER["target"].get(li)
        if heads:
            k=attn.shape[-1]; im=STEER["img_mask"]
            colmask=im[:k] if im.shape[0]>=k else torch.cat([im,torch.zeros(k-im.shape[0],dtype=torch.bool,device=im.device)])
            add=torch.zeros(attn.shape[1],device=attn.device,dtype=attn.dtype)
            for h in heads:
                if h<add.shape[0]: add[h]=STEER["alpha"]
            attn=attn+add[None,:,None,None]*colmask[None,None,None,:].to(attn.dtype)
    attn=torch.nn.functional.softmax(attn,dim=-1,dtype=torch.float32).to(query.dtype)
    attn=torch.nn.functional.dropout(attn,p=dropout,training=module.training)
    o=torch.matmul(attn,vs); return o.transpose(1,2).contiguous(), attn
qvl.eager_attention_forward=patched

def load(snap,dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return proc, Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev}, local_files_only=True).eval()

def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def ans_ids(proc):
    tok=proc.tokenizer
    f=lambda ws: list({tok.encode(w,add_special_tokens=False)[0] for w in ws})
    return f(["Same"," Same","same"," same"]), f(["Changed"," Changed","changed"," changed"])

@torch.no_grad()
def reason(proc,model,dev,img,q,mx=140):
    m=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+"\nThink briefly, then answer."}]}]
    t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(m)
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

@torch.no_grad()
def check(proc,model,dev,imgsw,q,R,cond,sid,cid):
    bu={"role":"user","content":[{"type":"image","image":imgsw},{"type":"text","text":q+"\nThink briefly, then answer."}]}
    if cond in ("self","self_steer"):
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE+CHECK+PRIME}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
    else:
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE+CHECK}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+PRIME.strip()
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":imgsw}]}])
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    STEER["img_mask"]=(inp["input_ids"][0]==model.config.image_token_id); STEER["on"]=(cond=="self_steer")
    out=model(**inp,use_cache=False); STEER["on"]=False
    lg=out.logits[0,-1].float()
    s=torch.logsumexp(lg[sid],0).item(); c=torch.logsumexp(lg[cid],0).item()
    return (1 if c>s else 0),(c-s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True); ap.add_argument("--prov_diff",required=True)
    ap.add_argument("--topk",type=int,default=60); ap.add_argument("--alpha",type=float,default=6.0)
    ap.add_argument("--n",type=int,default=40); ap.add_argument("--dev",default="cuda:0"); ap.add_argument("--out",default="/home/intern/swap2_out")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); STEER["alpha"]=a.alpha
    pd_=np.load(a.prov_diff); L,H=pd_.shape; top=np.argsort(pd_.flatten())[::-1][:a.topk]
    tgt={}; [tgt.setdefault(int(i//H),set()).add(int(i%H)) for i in top]; STEER["target"]=tgt
    print(f"proper operator: {a.topk} heads / {len(tgt)} layers, alpha={a.alpha}",flush=True)
    proc,model=load(a.snap,a.dev); sid,cid=ans_ids(proc); df=pd.read_parquet(a.parquet); n=min(a.n,len(df)); rows=[]
    for i in range(n):
        r=df.iloc[i]; rj=df.iloc[(i+1)%n]
        try: Ii=pil(r["image"]); Ij=pil(rj["image"]); q=str(r["question"])
        except: continue
        R=reason(proc,model,a.dev,Ii,q); rec={"i":i}
        for c in ("self","user","self_steer"):
            d,mg=check(proc,model,a.dev,Ij,q,R,c,sid,cid); rec[c]=d; rec[c+"_m"]=mg
        rows.append(rec)
        if (i+1)%5==0:
            dd=pd.DataFrame(rows); f=lambda c: dd[c].mean()
            print(f"[{i+1}/{n}] self={f('self'):.2f} user={f('user'):.2f} self+STEER={f('self_steer'):.2f}",flush=True)
    dd=pd.DataFrame(rows); dd.to_csv(os.path.join(a.out,"swap2.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(2000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== SWAP-DETECTION with PROPER operator, N={len(dd)}, alpha={a.alpha} ===")
    for c in ("self","user","self_steer"):
        m,lo,hi=boot(dd[c].values); print(f"  {c:11s} {m:.3f} [{lo:.3f},{hi:.3f}]")
    su,us,ss=dd["self"].mean(),dd["user"].mean(),dd["self_steer"].mean()
    print(f"\n  self->user gap = {us-su:+.3f}")
    print(f"  STEER recovery = {ss-su:+.3f} (want >0; toward/past user)")

if __name__=="__main__": main()
