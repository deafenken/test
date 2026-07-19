"""
FAITHFUL VisualSwap accuracy test (real semantic swap, extraction-free letter metric).
MathVerse answers are option letters. For each matched triplet (I_a, I_b, Q, A_a, A_b):
  base      : M(I_b, Q)                          -> should prefer A_b (sees the original image)
  self      : reason on I_a -> reflect(self) -> image swapped to I_b -> answer  (the illusion)
  prov_fix  : same reflection delivered as a USER turn (auto provenance re-frame; no human)
  self_steer: self + the per-head attention operator on localized provenance heads
Metric: 'prefers A_b' = logit(A_b letter) > logit(A_a letter). Illusion: self << base.
Recovery: prov_fix / self_steer > self, toward base.
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
ASK="\nAnswer with the option letter only."
PRIME="\nAnswer:"
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
def letter_id(proc, L):
    for w in (" "+L, L):
        t=proc.tokenizer.encode(w, add_special_tokens=False)
        if t: return t[0]
    return None

@torch.no_grad()
def reason(proc,model,dev,img,q,mx=140):
    m=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+ASK+"\nThink briefly first."}]}]
    t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(m)
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

@torch.no_grad()
def ans_logit(proc,model,dev,image,q,R,cond,aid,bid):
    bu={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q+ASK+"\nThink briefly first."}]}
    if cond=="base":
        m=[bu]; t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+PRIME.strip()
    elif cond in ("self","self_steer"):
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE+PRIME}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
    else:  # prov_fix (user turn)
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+PRIME.strip()
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":image}]}])
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    STEER["img_mask"]=(inp["input_ids"][0]==model.config.image_token_id); STEER["on"]=(cond=="self_steer")
    out=model(**inp,use_cache=False); STEER["on"]=False
    lg=out.logits[0,-1].float()
    return 1 if lg[bid].item()>lg[aid].item() else 0        # prefers A_b?

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True); ap.add_argument("--prov_diff",required=True)
    ap.add_argument("--topk",type=int,default=60); ap.add_argument("--alpha",type=float,default=6.0)
    ap.add_argument("--n",type=int,default=60); ap.add_argument("--dev",default="cuda:0"); ap.add_argument("--out",default="/home/intern/faithful_out")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); STEER["alpha"]=a.alpha
    pdm=np.load(a.prov_diff); L,H=pdm.shape; top=np.argsort(pdm.flatten())[::-1][:a.topk]
    tgt={}; [tgt.setdefault(int(i//H),set()).add(int(i%H)) for i in top]; STEER["target"]=tgt
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet); n=min(a.n,len(df)); rows=[]
    for i in range(n):
        r=df.iloc[i]
        try: Ia=pil(r["Ia"]); Ib=pil(r["Ib"]); q=str(r["question"]); Aa=str(r["A_a"]).strip(); Ab=str(r["A_b"]).strip()
        except Exception as e: print("skip",i,e); continue
        aid=letter_id(proc,Aa); bid=letter_id(proc,Ab)
        if aid is None or bid is None or aid==bid: continue
        R=reason(proc,model,a.dev,Ia,q)                       # reason on I_a
        rec={"i":i}
        rec["base"]=ans_logit(proc,model,a.dev,Ib,q,R,"base",aid,bid)     # sees I_b fresh
        for c in ("self","prov_fix","self_steer"):
            rec[c]=ans_logit(proc,model,a.dev,Ib,q,R,c,aid,bid)           # reasoned on I_a, image now I_b
        rows.append(rec)
        if (i+1)%5==0:
            d=pd.DataFrame(rows); f=lambda c:d[c].mean()
            print(f"[{i+1}/{n}] base={f('base'):.2f} self={f('self'):.2f} prov_fix={f('prov_fix'):.2f} self_steer={f('self_steer'):.2f}",flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,"faithful.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(2000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== FAITHFUL prefers-A_b rate, N={len(d)} (illusion: self<<base; recovery: fix>self) ===")
    for c in ("base","self","prov_fix","self_steer"):
        m,lo,hi=boot(d[c].values); print(f"  {c:11s} {m:.3f} [{lo:.3f},{hi:.3f}]")
    b,s=d["base"].mean(),d["self"].mean()
    print(f"\n  illusion gap (base-self) = {b-s:+.3f}")
    print(f"  prov_fix recovery = {d['prov_fix'].mean()-s:+.3f}")
    print(f"  operator recovery = {d['self_steer'].mean()-s:+.3f}")
    json.dump({c:list(boot(d[c].values)) for c in ('base','self','prov_fix','self_steer')}, open(os.path.join(a.out,'summary.json'),'w'), indent=1)

if __name__=="__main__": main()
