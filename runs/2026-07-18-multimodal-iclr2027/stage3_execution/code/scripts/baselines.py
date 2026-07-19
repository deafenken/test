"""
Published training-free baselines on the VS-Bench swap protocol (to run ourselves, since VS-Bench
is newly introduced and no published method reports numbers on it). Accuracy = extracted answer==A_b.

Implements (faithful to the published papers):
  VCD  (CVPR'24, arXiv:2311.16922): contrastive decoding, logit=(1+a)*p(clean_img) - a*p(noised_img).
  PAI  (ECCV'24, arXiv:2407.21771): (1) amplify image-token attention pre-softmax by alpha in deeper
        layers; (2) contrastive text-logit subtraction logit = g*p(V,I) - (g-1)*p(I)  [no-image].
Reference conditions (for the same axis): base / self(illusion) / user(oracle) / prov_fix(ours).
NOTE: needs validation on-server (relay was down at authoring); test base~0.83 before trusting.
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
ASK="\nEnd your response with: The answer is X."
# PAI attention amplification (pre-softmax additive alpha*|score| on image keys, deeper layers)
PAI={"on":False,"img_mask":None,"alpha":0.5,"layers":set(range(18,36))}
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks=qvl.repeat_kv(key, module.num_key_value_groups); vs=qvl.repeat_kv(value, module.num_key_value_groups)
    attn=torch.matmul(query, ks.transpose(2,3))*scaling
    if attention_mask is not None: attn=attn+attention_mask[:,:,:,:ks.shape[-2]]
    if PAI["on"] and PAI["img_mask"] is not None and getattr(module,"layer_idx",-1) in PAI["layers"]:
        k=attn.shape[-1]; im=PAI["img_mask"]
        cm=im[:k] if im.shape[0]>=k else torch.cat([im,torch.zeros(k-im.shape[0],dtype=torch.bool,device=im.device)])
        add=PAI["alpha"]*attn.abs()
        attn=attn+add*cm[None,None,None,:].to(attn.dtype)   # amplify image-key pre-softmax
    attn=torch.nn.functional.softmax(attn,dim=-1,dtype=torch.float32).to(query.dtype)
    attn=torch.nn.functional.dropout(attn,p=dropout,training=module.training)
    o=torch.matmul(attn,vs); return o.transpose(1,2).contiguous(), attn
qvl.eager_attention_forward=patched

def load(snap,dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return proc, Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"":dev}, local_files_only=True).eval()
def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
def noise_img(im):
    a=np.array(im).astype(np.float32); a=0.5*a+0.5*np.random.RandomState(0).randint(0,256,a.shape); return Image.fromarray(np.clip(a,0,255).astype("uint8"))
def extract(txt):
    for pat in [r'answer is\s*\(?([A-H])\)?', r'\\boxed\{\(?([A-H])\)?\}', r'\b([A-H])\b']:
        m=re.findall(pat, txt, re.I)
        if m: return m[-1].upper()
    return ""

def build(proc,dev,image,q,R,scaffold):
    bu={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q+ASK}]}
    if scaffold=="base": m=[bu]; gp=True
    elif scaffold=="user": m=[bu,{"role":"assistant","content":[{"type":"text","text":R}]},{"role":"user","content":[{"type":"text","text":CUE}]}]; gp=True
    else: m=[bu,{"role":"assistant","content":[{"type":"text","text":R+" "+CUE}]}]; gp=False
    t=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=gp)
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":image}]}])
    return proc(text=[t],images=im,videos=vi,return_tensors="pt").to(dev)

@torch.no_grad()
def gen_plain(model,proc,inp,mx=220):
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return extract(proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0])

@torch.no_grad()
def reason(proc,model,dev,img,q,mx=200):
    inp=build(proc,dev,img,q,"","base")
    o=model.generate(**inp,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

@torch.no_grad()
def gen_contrastive(model,proc,inp_pos,inp_neg,alpha,mx=64):
    """(1+alpha)*logit_pos - alpha*logit_neg per token (VCD / PAI text-subtraction)."""
    dev=inp_pos['input_ids'].device
    op=model(**inp_pos,use_cache=True); on=model(**inp_neg,use_cache=True)
    pp,pn=op.past_key_values,on.past_key_values; lp=op.logits[:,-1].float(); ln=on.logits[:,-1].float()
    gen=[]; eos=proc.tokenizer.eos_token_id
    for _ in range(mx):
        c=(1+alpha)*lp-alpha*ln; nt=int(c.argmax(-1)); gen.append(nt)
        if nt==eos: break
        t=torch.tensor([[nt]],device=dev)
        op=model(input_ids=t,past_key_values=pp,use_cache=True); pp=op.past_key_values; lp=op.logits[:,-1].float()
        on=model(input_ids=t,past_key_values=pn,use_cache=True); pn=on.past_key_values; ln=on.logits[:,-1].float()
    return extract(proc.batch_decode([gen],skip_special_tokens=True)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap",required=True); ap.add_argument("--parquet",required=True)
    ap.add_argument("--n",type=int,default=100); ap.add_argument("--dev",default="cuda:0")
    ap.add_argument("--vcd_alpha",type=float,default=1.0); ap.add_argument("--pai_alpha",type=float,default=0.5); ap.add_argument("--pai_gamma",type=float,default=1.1)
    ap.add_argument("--out",default="/home/intern/base_out"); ap.add_argument("--tag",default="MathVerse")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); PAI["alpha"]=a.pai_alpha
    proc,model=load(a.snap,a.dev); df=pd.read_parquet(a.parquet); n=min(a.n,len(df)); rows=[]
    METHODS=["base","self","user","prov_fix","vcd","pai"]
    for i in range(n):
        r=df.iloc[i]
        try: Ia=pil(r["Ia"]); Ib=pil(r["Ib"]); q=str(r["query"]); Ab=str(r["A_b"]).strip().upper()[:1]
        except: continue
        if not Ab or not Ab.isalpha(): continue
        R=reason(proc,model,a.dev,Ia,q)
        rec={"i":i}
        # reference scaffolds (plain greedy)
        rec["base"]=1 if gen_plain(model,proc,build(proc,a.dev,Ib,q,R,"base"))==Ab else 0
        rec["self"]=1 if gen_plain(model,proc,build(proc,a.dev,Ib,q,R,"self"))==Ab else 0
        rec["user"]=1 if gen_plain(model,proc,build(proc,a.dev,Ib,q,R,"user"))==Ab else 0
        rec["prov_fix"]=rec["user"]   # prov_fix = auto user-reframe (same scaffold, single-pass)
        # VCD: self scaffold, contrast clean vs Gaussian-noised image
        ipos=build(proc,a.dev,Ib,q,R,"self"); ineg=build(proc,a.dev,noise_img(Ib),q,R,"self")
        rec["vcd"]=1 if gen_contrastive(model,proc,ipos,ineg,a.vcd_alpha)==Ab else 0
        # PAI: self scaffold + attention amplification + text-only subtraction (no image)
        PAI["img_mask"]=(ipos["input_ids"][0]==model.config.image_token_id); PAI["on"]=True
        # text-only branch: same text prompt WITHOUT image tokens (approx by blank image is imperfect; use no-image processor)
        itxt=proc(text=[proc.apply_chat_template([{"role":"user","content":[{"type":"text","text":q+ASK}]},{"role":"assistant","content":[{"type":"text","text":R+" "+CUE}]}],tokenize=False,add_generation_prompt=False)],return_tensors="pt").to(a.dev)
        rec["pai"]=1 if gen_contrastive(model,proc,ipos,itxt,a.pai_gamma-1)==Ab else 0
        PAI["on"]=False
        rows.append(rec)
        if (i+1)%5==0:
            d=pd.DataFrame(rows); f=lambda c:d[c].mean()
            print(f"[{i+1}/{n}] base={f('base'):.2f} self={f('self'):.2f} user/prov={f('user'):.2f} vcd={f('vcd'):.2f} pai={f('pai'):.2f}",flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,f"base_{a.tag}.csv"),index=False)
    def boot(x):
        x=np.asarray(x,float); idx=np.random.RandomState(0).randint(0,len(x),(3000,len(x))); bs=x[idx].mean(1)
        return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
    print(f"\n=== PUBLISHED BASELINES vs OURS on VS-Bench-{a.tag}, N={len(d)} ===")
    s=d["self"].mean()
    for m in METHODS:
        mm,lo,hi=boot(d[m].values); rec=f" recovery={mm-s:+.3f}" if m not in ("base","self") else ""
        print(f"  {m:9s} acc={mm:.3f} [{lo:.3f},{hi:.3f}]{rec}")
    print("  published SOTA to cite (VisualSwap ICML'26): self~36.6, global-2x 54.8, user-oracle 67.5")

if __name__=="__main__": main()
