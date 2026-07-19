"""
SOTA comparison on the faithful VisualSwap accuracy metric.
Training-free attention-steering baselines vs our provenance-injection fix.
Intervention = post-softmax amplify-and-renormalize of image-key attention by factor f (the
phenomenon paper's / PAI-family method), applied to a target head-set:
  global_2x   : ALL heads, f=2   (arXiv:2605.15864's own fix)
  cast_style  : top-K image-attending heads, f=2   (approximates CAST/DMAS head-subset steering)
  ours_op     : our localized provenance heads, f=2 (our attention operator)
  prov_fix    : NO attention intervention; auto-reframe self-reflection as a user turn (OURS)
Metric: prefers-A_b (logit(A_b letter) > logit(A_a)). Illusion: self<<base. A method 'works' if > self.
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
ASK="\nAnswer with the option letter only."; PRIME="\nAnswer:"
# STEER: mode multiply post-softmax; target None=off, 'all'=all heads, dict=layer->set(heads)
STEER={"on":False,"img_mask":None,"factor":2.0,"target":None}

def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks=qvl.repeat_kv(key, module.num_key_value_groups); vs=qvl.repeat_kv(value, module.num_key_value_groups)
    attn=torch.matmul(query, ks.transpose(2,3))*scaling
    if attention_mask is not None: attn=attn+attention_mask[:,:,:,:ks.shape[-2]]
    attn=torch.nn.functional.softmax(attn,dim=-1,dtype=torch.float32).to(query.dtype)
    if STEER["on"] and STEER["img_mask"] is not None and STEER["target"] is not None:
        li=getattr(module,"layer_idx",None)
        tgt=STEER["target"]
        heads = range(attn.shape[1]) if tgt=="all" else tgt.get(li, [])
        if len(heads):
            k=attn.shape[-1]; im=STEER["img_mask"]
            cm=im[:k] if im.shape[0]>=k else torch.cat([im,torch.zeros(k-im.shape[0],dtype=torch.bool,device=im.device)])
            f=STEER["factor"]
            a=attn.clone()
            for h in heads:
                if h>=a.shape[1]: continue
                col=a[:,h,:,:].clone()
                col[:,:,cm]=col[:,:,cm]*f
                col=col/col.sum(-1,keepdim=True).clamp_min(1e-9)
                a[:,h,:,:]=col
            attn=a
    attn=torch.nn.functional.dropout(attn,p=dropout,training=module.training)
    o=torch.matmul(attn,vs); return o.transpose(1,2).contiguous(), attn
qvl.eager_attention_forward=patched

def load(snap,dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return proc, Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev}, local_files_only=True).eval()
def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def lid(proc,L):
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
def probe(proc,model,dev,image,q,R,method,aid,bid,heads_topk):
    """method in: base, self, prov_fix, global_2x, cast_style, ours_op"""
    bu={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q+ASK+"\nThink briefly first."}]}
    if method=="base":
        m=[bu]; t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+PRIME.strip()
    elif method=="prov_fix":
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+PRIME.strip()
    else:  # self and all attention-steering baselines: self-reflection continuation
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE+PRIME}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":image}]}])
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    STEER["img_mask"]=(inp["input_ids"][0]==model.config.image_token_id)
    STEER["target"]={"self":None,"base":None,"prov_fix":None,
                     "global_2x":"all","cast_style":heads_topk,"ours_op":heads_topk}[method]
    STEER["on"]=method in ("global_2x","cast_style","ours_op"); STEER["factor"]=2.0
    out=model(**inp,use_cache=False); STEER["on"]=False
    lg=out.logits[0,-1].float()
    return 1 if lg[bid].item()>lg[aid].item() else 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True); ap.add_argument("--prov_diff",required=True)
    ap.add_argument("--topk",type=int,default=60); ap.add_argument("--n",type=int,default=200); ap.add_argument("--dev",default="cuda:0")
    ap.add_argument("--out",default="/home/intern/sota_out"); ap.add_argument("--tag",default="MathVerse")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    pdm=np.load(a.prov_diff); L,H=pdm.shape; top=np.argsort(pdm.flatten())[::-1][:a.topk]
    heads_topk={}; [heads_topk.setdefault(int(i//H),set()).add(int(i%H)) for i in top]
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet); n=min(a.n,len(df))
    METHODS=["base","self","global_2x","cast_style","ours_op","prov_fix"]
    rows=[]
    for i in range(n):
        r=df.iloc[i]
        try: Ia=pil(r["Ia"]); Ib=pil(r["Ib"]); q=str(r["question"]); Aa=str(r["A_a"]).strip(); Ab=str(r["A_b"]).strip()
        except: continue
        aid=lid(proc,Aa); bid=lid(proc,Ab)
        if aid is None or bid is None or aid==bid: continue
        R=reason(proc,model,a.dev,Ia,q)
        rec={"i":i}
        for mth in METHODS:
            img = Ib   # all probes show the swapped-to original image I_b
            rec[mth]=probe(proc,model,a.dev,img,q,R,mth,aid,bid,heads_topk)
        rows.append(rec)
        if (i+1)%10==0:
            d=pd.DataFrame(rows); f=lambda c:d[c].mean()
            print(f"[{i+1}/{n}] base={f('base'):.2f} self={f('self'):.2f} g2x={f('global_2x'):.2f} cast={f('cast_style'):.2f} ours_op={f('ours_op'):.2f} prov_fix={f('prov_fix'):.2f}",flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,f"sota_{a.tag}.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(3000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== SOTA COMPARISON on VisualSwap-{a.tag}, N={len(d)} (prefers-A_b; higher=better; self=illusion floor, base=ceiling) ===")
    s=d["self"].mean()
    for m in METHODS:
        mm,lo,hi=boot(d[m].values); rec=f" recovery={mm-s:+.3f}" if m not in ("base","self") else ""
        print(f"  {m:11s} {mm:.3f} [{lo:.3f},{hi:.3f}]{rec}")
    json.dump({m:list(boot(d[m].values)) for m in METHODS}, open(os.path.join(a.out,f"summary_{a.tag}.json"),"w"), indent=1)

if __name__=="__main__": main()
