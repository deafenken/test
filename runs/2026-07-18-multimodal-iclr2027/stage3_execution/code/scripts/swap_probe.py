"""
Behavioral illusion + recovery + fix (self-contained, real VisualSwap images).

For each instance: model reasons on image I_i (question Q_i). At the re-look step the image
is SWAPPED to a different image I_j. We ask a matched Yes/No consistency check under:
  self   : re-look cue delivered as the assistant's own continuation
  user   : the SAME cue delivered as a USER turn
  self+STEER : self condition, but our training-free steering amplifies image attention on
               the localized layers at the check step (the FIX)
Since the image WAS swapped, the correct answer is "No" (it changed). Detection rate =
fraction answering No. Illusion => self detection low; user detection high; STEER recovers self.
"""
import os, sys, json, argparse, re
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch, numpy as np, pandas as pd
from PIL import Image
import io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

CUE = "Wait, let me check the figure again to make sure I haven't made a mistake."
CHECK = " Look carefully at the image shown right now. Is it the SAME figure described in the analysis above, or has it changed? Answer with one word: Same or Changed."
STEER_LAYERS = (3, 6, 12, 14, 22)   # layers hosting top provenance heads (from N=100 localization)

def load(snap, dev):
    proc=AutoProcessor.from_pretrained(snap, local_files_only=True)
    arch=json.load(open(os.path.join(snap,"config.json")))["architectures"][0]
    from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    Cls=Qwen3VLForConditionalGeneration if ("Qwen3VL" in arch or "Qwen3_VL" in arch) else Qwen2_5_VLForConditionalGeneration
    m=Cls.from_pretrained(snap, torch_dtype=torch.bfloat16, attn_implementation="eager",
                          device_map={"":dev}, local_files_only=True).eval()
    return proc, m

def pil(v): return Image.open(io.BytesIO(v["bytes"])).convert("RGB")

def answer_token_ids(proc):
    tok=proc.tokenizer
    def ids(words):
        s=set()
        for w in words:
            for t in tok.encode(w, add_special_tokens=False)[:1]: s.add(t)
        return list(s)
    return ids(["Same"," Same","same"," same"]), ids(["Changed"," Changed","changed"," changed"])

STEER={"on":False,"scale":1.0,"mask":None}
def install_hooks(model):
    layers=model.model.language_model.layers if hasattr(model.model,"language_model") else model.model.layers
    hs=[]
    def mk():
        def pre(module, args, kwargs):
            h=kwargs.get("hidden_states", args[0] if args else None)
            # apply ONLY at prefill (h seq-len == prompt len == mask len); scales image key
            # states into the KV cache, amplifying visual attention for all later decode steps.
            if (h is not None and STEER["on"] and STEER["mask"] is not None
                    and h.shape[1]==STEER["mask"].shape[0]):
                h=h.clone(); h[0, STEER["mask"], :]*=STEER["scale"]
                if "hidden_states" in kwargs: kwargs["hidden_states"]=h
                else: args=(h,)+tuple(args[1:])
            return args, kwargs
        return pre
    for li in STEER_LAYERS:
        hs.append(layers[li].self_attn.register_forward_pre_hook(mk(), with_kwargs=True))
    return hs

@torch.no_grad()
def reason(proc, model, dev, img, q, max_new=160):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q+"\nThink briefly, then give the final answer."}]}]
    text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im,vi=process_vision_info(msgs); inp=proc(text=[text], images=im, videos=vi, return_tensors="pt").to(dev)
    out=model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:,inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

PRIME = "\nMy one-word answer (Same or Changed) is:"
@torch.no_grad()
def check(proc, model, dev, img_swapped, q, R, cond, same_ids, changed_ids):
    base_user={"role":"user","content":[{"type":"image","image":img_swapped},{"type":"text","text":q+"\nThink briefly, then give the final answer."}]}
    if cond in ("self","self_steer"):
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R+" "+CUE+CHECK+PRIME}]}]
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    else:  # user
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R}]},
              {"role":"user","content":[{"type":"text","text":CUE+CHECK}]}]
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)+PRIME.strip()
    im,vi=process_vision_info([{"role":"user","content":[{"type":"image","image":img_swapped}]}])
    inp=proc(text=[text], images=im, videos=vi, return_tensors="pt").to(dev)
    STEER["mask"]=(inp["input_ids"][0]==model.config.image_token_id)
    STEER["on"]=(cond=="self_steer")
    out=model(**inp, use_cache=False)          # single forward; steering applies at prefill
    STEER["on"]=False
    lg=out.logits[0,-1].float()
    s=torch.logsumexp(lg[same_ids],0).item(); c=torch.logsumexp(lg[changed_ids],0).item()
    return (1 if c>s else 0), (c-s)            # detected(Changed)? , margin

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--scale", type=float, default=1.6); ap.add_argument("--out", default="/home/intern/swap_out")
    a=ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    STEER["scale"]=a.scale
    proc, model = load(a.snap, a.dev); install_hooks(model)
    same_ids, changed_ids = answer_token_ids(proc)
    print("same_ids",same_ids,"changed_ids",changed_ids, flush=True)
    df=pd.read_parquet(a.parquet); n=min(a.n, len(df))
    rows=[]
    for i in range(n):
        r=df.iloc[i]; rj=df.iloc[(i+1)%n]        # swap to the NEXT image (different figure)
        try: I_i=pil(r["image"]); I_j=pil(rj["image"]); q=str(r["question"])
        except Exception as e: print("skip",i,e); continue
        R=reason(proc, model, a.dev, I_i, q)
        rec={"i":i}
        for cond in ("self","user","self_steer"):
            det,margin=check(proc, model, a.dev, I_j, q, R, cond, same_ids, changed_ids)
            rec[cond]=det; rec[cond+"_margin"]=margin
        rows.append(rec)
        if (i+1)%5==0:
            d=pd.DataFrame(rows)
            det=lambda c: d[c].dropna().mean()
            print(f"[{i+1}/{n}] detect(Changed) self={det('self'):.2f} user={det('user'):.2f} self+STEER={det('self_steer'):.2f}", flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(a.out,"swap.csv"), index=False)
    def boot(x):
        x=np.asarray(x,float); x=x[~np.isnan(x)]
        if len(x)==0: return (float("nan"),)*3
        idx=np.random.RandomState(0).randint(0,len(x),(2000,len(x))); bs=x[idx].mean(1)
        return x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
    print(f"\n=== SWAP-DETECTION (Changed) rate, N={len(d)} ===")
    for c in ("self","user","self_steer"):
        m,lo,hi=boot(d[c].values); print(f"  {c:11s} {m:.3f} [{lo:.3f},{hi:.3f}]  (n_valid={d[c].notna().sum()})")
    # recovery: does STEER close the self->user gap?
    su=np.asarray(d["self"].values,float); us=np.asarray(d["user"].values,float); ss=np.asarray(d["self_steer"].values,float)
    print(f"\n  self->user gap = {np.nanmean(us)-np.nanmean(su):+.3f}")
    print(f"  STEER recovery = {np.nanmean(ss)-np.nanmean(su):+.3f} (want >0, toward user)")
    json.dump({c:list(boot(d[c].values)) for c in ("self","user","self_steer")},
              open(os.path.join(a.out,"summary.json"),"w"), indent=1)

if __name__=="__main__": main()
