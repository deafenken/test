"""
PACD — Provenance-Contrastive Decoding (the pursue-rated novel innovation).
Single-pass (per token, two forward branches) contrastive decoding over IDENTICAL image I_b and
IDENTICAL re-look content, differing ONLY in role: z_user (re-look as a user turn) vs z_self
(assistant continuation). Decode from softmax((1+λ)·z_user − λ·z_self).
Compares: base, self (illusion), user (ceiling), prov_fix (=user branch greedy), pacd.
Accuracy = generated final answer letter == A_b (MathVerse letter subset).
"""
import os, json, argparse, re
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, numpy as np, pandas as pd
from PIL import Image
import io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

CUE="Wait, let me check the figure again to make sure I haven't made a mistake."
ASK="\nAnswer with the option letter only."
FINAL="\nTherefore, the final answer (letter) is:"

def load(snap,dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return proc, Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev}, local_files_only=True).eval()
def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def extract_letter(txt):
    m=re.findall(r'\b([A-H])\b', txt.upper()); return m[-1] if m else (txt.strip().upper()[:1] if txt.strip() else "")

@torch.no_grad()
def reason(proc,model,dev,img,q,mx=140):
    m=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+ASK+"\nThink briefly first."}]}]
    t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(m)
    inp=proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

def build_inputs(proc,dev,image,q,R,branch):
    bu={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q+ASK+"\nThink briefly first."}]}
    if branch=="base":
        m=[bu]; t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+FINAL.strip()
    elif branch=="user":
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)+FINAL.strip()
    else:  # self
        m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE+FINAL}]}]
        t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":image}]}])
    return proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)

@torch.no_grad()
def greedy(model,inp,steps=8):
    out=model.generate(**inp,max_new_tokens=steps,do_sample=False)
    return out[:,inp['input_ids'].shape[1]:]

@torch.no_grad()
def pacd_decode(model,proc,inp_user,inp_self,lam=1.0,steps=8):
    """Contrastive: next = argmax((1+lam)*logit_user - lam*logit_self). Append token to BOTH branches."""
    dev=inp_user['input_ids'].device
    ou=model(**inp_user,use_cache=True); os_=model(**inp_self,use_cache=True)
    pku=ou.past_key_values; pks=os_.past_key_values
    lu=ou.logits[:,-1,:].float(); ls=os_.logits[:,-1,:].float()
    gen=[]
    eos=model.config.eos_token_id if isinstance(model.config.eos_token_id,int) else (model.config.eos_token_id[0] if model.config.eos_token_id else None)
    for _ in range(steps):
        comb=(1+lam)*lu - lam*ls
        nt=int(comb.argmax(-1).item()); gen.append(nt)
        if eos is not None and nt==eos: break
        tid=torch.tensor([[nt]],device=dev)
        ou=model(input_ids=tid,past_key_values=pku,use_cache=True); pku=ou.past_key_values; lu=ou.logits[:,-1,:].float()
        os_=model(input_ids=tid,past_key_values=pks,use_cache=True); pks=os_.past_key_values; ls=os_.logits[:,-1,:].float()
    return proc.batch_decode([gen],skip_special_tokens=True)[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True)
    ap.add_argument("--lam",type=float,default=1.0); ap.add_argument("--n",type=int,default=120); ap.add_argument("--dev",default="cuda:0")
    ap.add_argument("--out",default="/home/intern/pacd_out"); ap.add_argument("--tag",default="MathVerse")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet); n=min(a.n,len(df)); rows=[]
    for i in range(n):
        r=df.iloc[i]
        try: Ia=pil(r["Ia"]); Ib=pil(r["Ib"]); q=str(r["question"]); Ab=str(r["A_b"]).strip().upper()[:1]
        except: continue
        if not Ab or not Ab.isalpha(): continue
        R=reason(proc,model,a.dev,Ia,q)
        iu=build_inputs(proc,a.dev,Ib,q,R,"user"); iss=build_inputs(proc,a.dev,Ib,q,R,"self"); ib=build_inputs(proc,a.dev,Ib,q,R,"base")
        rec={"i":i,"Ab":Ab}
        rec["base"]=1 if extract_letter(proc.batch_decode(greedy(model,ib),skip_special_tokens=True)[0])==Ab else 0
        rec["self"]=1 if extract_letter(proc.batch_decode(greedy(model,iss),skip_special_tokens=True)[0])==Ab else 0
        rec["user"]=1 if extract_letter(proc.batch_decode(greedy(model,iu),skip_special_tokens=True)[0])==Ab else 0
        rec["pacd"]=1 if extract_letter(pacd_decode(model,proc,iu,iss,lam=a.lam))==Ab else 0
        rows.append(rec)
        if (i+1)%10==0:
            d=pd.DataFrame(rows); f=lambda c:d[c].mean()
            print(f"[{i+1}/{n}] base={f('base'):.2f} self={f('self'):.2f} user={f('user'):.2f} pacd={f('pacd'):.2f}",flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,f"pacd_{a.tag}.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(3000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== PACD (lam={a.lam}) accuracy on {a.tag}, N={len(d)} ===")
    s=d["self"].mean()
    for m in ["base","self","user","pacd"]:
        mm,lo,hi=boot(d[m].values); rec=f" recovery={mm-s:+.3f}" if m not in ("base","self") else ""
        print(f"  {m:6s} acc={mm:.3f} [{lo:.3f},{hi:.3f}]{rec}")

if __name__=="__main__": main()
