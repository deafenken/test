"""
Protocol-matched VS-Bench reproduction: full R_b generation (not a forced short letter) + robust
answer extraction, to place our operating point at the paper's scale (base~0.83, probe~0.30) and
let global-2x show its real recovery. Then prov_fix / mag-match on the SAME axis (the valid money plot).

Methods: base / self(Probe) / user(Multi-turn) / prov_fix / global_2x / mag_match.
global_2x & mag_match apply attention amplification DURING the full R_b generation.
Accuracy = extracted final answer letter == A_b.
"""
import os, json, argparse, re
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, numpy as np, pandas as pd
from PIL import Image
import io
from transformers import AutoProcessor
import transformers.models.qwen3_vl.modeling_qwen3_vl as qvl
from qwen_vl_utils import process_vision_info

CUE="Wait, let me check the figure again to make sure I haven't made a mistake."
ASK="\nAnswer with the option letter. End with: The answer is X."
STEER={"on":False,"img_mask":None,"factor":2.0,"target":None}

def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks=qvl.repeat_kv(key, module.num_key_value_groups); vs=qvl.repeat_kv(value, module.num_key_value_groups)
    attn=torch.matmul(query, ks.transpose(2,3))*scaling
    if attention_mask is not None: attn=attn+attention_mask[:,:,:,:ks.shape[-2]]
    attn=torch.nn.functional.softmax(attn,dim=-1,dtype=torch.float32).to(query.dtype)
    if STEER["on"] and STEER["img_mask"] is not None and STEER["target"] is not None:
        li=getattr(module,"layer_idx",None); tgt=STEER["target"]
        heads=range(attn.shape[1]) if tgt=="all" else tgt.get(li,[])
        if len(heads):
            k=attn.shape[-1]; im=STEER["img_mask"]
            cm=im[:k] if im.shape[0]>=k else torch.cat([im,torch.zeros(k-im.shape[0],dtype=torch.bool,device=im.device)])
            f=STEER["factor"]; a=attn.clone()
            for h in heads:
                if h>=a.shape[1]: continue
                col=a[:,h,:,:].clone(); col[:,:,cm]=col[:,:,cm]*f
                col=col/col.sum(-1,keepdim=True).clamp_min(1e-9); a[:,h,:,:]=col
            attn=a
    attn=torch.nn.functional.dropout(attn,p=dropout,training=module.training)
    o=torch.matmul(attn,vs); return o.transpose(1,2).contiguous(), attn
qvl.eager_attention_forward=patched

def load(snap,dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return proc, Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev}, local_files_only=True).eval()
def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def extract(txt):
    for pat in [r'answer is\s*\(?([A-H])\)?', r'\\boxed\{\(?([A-H])\)?\}', r'\b([A-H])\b']:
        m=re.findall(pat, txt, re.I)
        if m: return m[-1].upper()
    return ""

@torch.no_grad()
def reason(proc,model,dev,img,q,mx=200):
    m=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+ASK}]}]
    t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(m)
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

@torch.no_grad()
def answer(proc,model,dev,image,q,R,method,heads_topk,mag_factor,mx=220):
    bu={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q+ASK}]}
    if method=="base":
        m=[bu]; t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    elif method in ("user","prov_fix"):
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    else:  # self / global_2x / mag_match : self-reflection continuation, full R_b
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":image}]}])
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    STEER["img_mask"]=(inp["input_ids"][0]==model.config.image_token_id)
    if method=="global_2x": STEER["target"]="all"; STEER["factor"]=2.0; STEER["on"]=True
    elif method=="mag_match": STEER["target"]=heads_topk; STEER["factor"]=mag_factor; STEER["on"]=True
    else: STEER["on"]=False
    out=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    STEER["on"]=False
    return extract(proc.batch_decode(out[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True); ap.add_argument("--prov_diff",required=True)
    ap.add_argument("--topk",type=int,default=60); ap.add_argument("--mag_factor",type=float,default=6.0)
    ap.add_argument("--n",type=int,default=100); ap.add_argument("--dev",default="cuda:0"); ap.add_argument("--out",default="/home/intern/vsfull_out"); ap.add_argument("--tag",default="MathVerse")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    pdm=np.load(a.prov_diff); L,H=pdm.shape; top=np.argsort(pdm.flatten())[::-1][:a.topk]
    heads_topk={}; [heads_topk.setdefault(int(i//H),set()).add(int(i%H)) for i in top]
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet); n=min(a.n,len(df)); rows=[]
    METHODS=["base","self","user","prov_fix","global_2x","mag_match"]
    for i in range(n):
        r=df.iloc[i]
        try: Ia=pil(r["Ia"]); Ib=pil(r["Ib"]); q=str(r["question"]); Ab=str(r["A_b"]).strip().upper()[:1]
        except: continue
        if not Ab or not Ab.isalpha(): continue
        R=reason(proc,model,a.dev,Ia,q)
        rec={"i":i,"Ab":Ab}
        for mth in METHODS:
            rec[mth]=1 if answer(proc,model,a.dev,Ib,q,R,mth,heads_topk,a.mag_factor)==Ab else 0
        rows.append(rec)
        if (i+1)%5==0:
            d=pd.DataFrame(rows); f=lambda c:d[c].mean()
            print(f"[{i+1}/{n}] base={f('base'):.2f} self={f('self'):.2f} user={f('user'):.2f} prov={f('prov_fix'):.2f} g2x={f('global_2x'):.2f} magm={f('mag_match'):.2f}",flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,f"vsfull_{a.tag}.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(3000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== PROTOCOL-MATCHED VS-BENCH (full R_b) on {a.tag}, N={len(d)} ===")
    s=d["self"].mean()
    for m in METHODS:
        mm,lo,hi=boot(d[m].values); rec=f" recovery={mm-s:+.3f}" if m not in ("base","self") else ""
        print(f"  {m:11s} acc={mm:.3f} [{lo:.3f},{hi:.3f}]{rec}")
    print("  (target: base~0.83, self/probe~0.30, global_2x recovery~+0.18 to reproduce the paper)")

if __name__=="__main__": main()
