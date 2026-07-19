"""
Week-1 mechanism experiment (P1): does the visual-attention surge at a re-look
span depend on the trigger's PROVENANCE (user vs self) vs turn-BOUNDARY vs marker/sink?

Matched-content design on real VisualSwap images (I_a). Byte-identical reflection
string P across conditions; image content fixed. Measures per-head visual-attention
mass S_vis during the re-look continuation.

Conditions (all continue AFTER the model's own reasoning R on (image, Q)):
  C1  self / mid-decode      : P appended inside the SAME assistant turn (the illusion baseline)
  C2  user / far-back        : P delivered as a USER turn (image NOT re-inserted) -> isolates PROVENANCE
  C3  assistant / boundary   : P starts a NEW assistant turn (boundary, self role) -> isolates BOUNDARY
  C0m marker / empty         : user turn with EMPTY/neutral content -> isolates marker/sink insertion

H1: S_vis(C2) > S_vis(C1)  (provenance surge), C3 ~= C1 (boundary alone insufficient),
    C0m ~= C1 (markers alone insufficient).
"""
import os, sys, json, argparse, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch, numpy as np, pandas as pd
from PIL import Image
import io, json as _json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

P = "Wait, let me check the figure again to make sure I haven't made a mistake."
P_EMPTY = "OK."   # neutral marker-only content for C0m

def load_model(snap, dev):
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    arch = _json.load(open(os.path.join(snap,"config.json"))).get("architectures",[""])[0]
    if "Qwen3VL" in arch or "Qwen3_VL" in arch:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    print(f"model class: {Cls.__name__} (arch={arch})", flush=True)
    model = Cls.from_pretrained(
        snap, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"":dev}, local_files_only=True)
    model.eval()
    return proc, model

def pil_from_field(v):
    if isinstance(v, dict) and "bytes" in v and v["bytes"] is not None:
        return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    if isinstance(v, dict) and v.get("path"):
        return Image.open(v["path"]).convert("RGB")
    raise ValueError("bad image field")

def img_mask_for(inputs, model):
    ids = inputs["input_ids"][0]
    return (ids == model.config.image_token_id)

@torch.no_grad()
def gen_reasoning(proc, model, dev, img, q, max_new=96):
    msgs=[{"role":"user","content":[{"type":"image","image":img},
           {"type":"text","text":q+"\nThink briefly, then give the final answer."}]}]
    text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    inputs=proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(dev)
    out=model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    R=proc.batch_decode(out[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return msgs, R

def build_condition(proc, img, q, R, cond):
    """Return chat `text` (for the processor) with the re-look trigger placed per condition.
    We measure attention during the assistant continuation AFTER this text."""
    base_user={"role":"user","content":[{"type":"image","image":img},
               {"type":"text","text":q+"\nThink briefly, then give the final answer."}]}
    if cond=="C1":   # self / mid-decode: reflection inside the same assistant turn, continue it
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R+" "+P}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if cond=="C3":   # assistant boundary: R closes; a NEW assistant turn holds P
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R}]},
              {"role":"assistant","content":[{"type":"text","text":P}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if cond=="C2":   # user turn: R closes; USER issues P; assistant will respond
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R}]},
              {"role":"user","content":[{"type":"text","text":P}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if cond=="C0m":  # marker/sink control: user turn with neutral content
        msgs=[base_user, {"role":"assistant","content":[{"type":"text","text":R}]},
              {"role":"user","content":[{"type":"text","text":P_EMPTY}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raise ValueError(cond)

@torch.no_grad()
def measure_svis(proc, model, dev, img, text, n_steps=8):
    """Generate n_steps tokens from `text`; return per-head visual mass averaged over
    generated steps: array [L, H]. Also return scalar mean over heads."""
    imgs,vids=process_vision_info([{"role":"user","content":[{"type":"image","image":img}]}])
    inputs=proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(dev)
    im_mask=img_mask_for(inputs, model)
    out=model.generate(**inputs, max_new_tokens=n_steps, do_sample=False,
                       output_attentions=True, return_dict_in_generate=True)
    # out.attentions: tuple over generated steps; each is tuple over layers of (b, H, q, kv)
    L=len(out.attentions[0]); H=out.attentions[0][0].shape[1]
    n_img=int(im_mask.sum().item())
    acc=np.zeros((L,H), dtype=np.float64); cnt=0
    kv_img = im_mask  # image tokens are in the prompt prefix; kv index aligns for step 0
    for step_att in out.attentions:
        for li in range(L):
            a=step_att[li][0,:,-1,:]              # (H, kv) attention of the newly generated token
            m=a[:, :im_mask.shape[0]][:, im_mask].sum(-1)  # visual mass per head (image keys in prefix)
            acc[li]+=m.float().cpu().numpy()
        cnt+=1
    perhead = acc/max(cnt,1)                       # [L,H]
    return perhead, float(perhead.mean()), n_img

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--out", default="/home/intern/provlook_out")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    proc, model = load_model(args.snap, args.dev)
    df=pd.read_parquet(args.parquet)
    qcol="question"
    rows=[]
    t0=time.time()
    conds=["C1","C2","C3","C0m"]
    for i in range(min(args.n, len(df))):
        r=df.iloc[i]
        try:
            img=pil_from_field(r["image"]); q=str(r[qcol])
        except Exception as e:
            print("skip",i,e); continue
        _, R = gen_reasoning(proc, model, args.dev, img, q)
        rec={"idx":int(i)}
        perheads={}
        for c in conds:
            text=build_condition(proc, img, q, R, c)
            ph, sv, n_img = measure_svis(proc, model, args.dev, img, text)
            rec[f"svis_{c}"]=sv; rec["n_img"]=n_img
            perheads[c]=ph
        rows.append(rec)
        # save per-head arrays for localization (C1 & C2)
        np.savez_compressed(os.path.join(args.out, f"perhead_{i:04d}.npz"),
                            **{c:perheads[c] for c in conds})
        if (i+1)%5==0:
            d=pd.DataFrame(rows)
            print(f"[{i+1}/{args.n} {time.time()-t0:.0f}s] "
                  f"C1={d.svis_C1.mean():.4f} C2={d.svis_C2.mean():.4f} "
                  f"C3={d.svis_C3.mean():.4f} C0m={d.svis_C0m.mean():.4f}", flush=True)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(args.out,"svis.csv"), index=False)
    # bootstrap CIs on contrasts
    def boot(x, nb=2000):
        x=np.asarray(x); idx=np.random.RandomState(0).randint(0,len(x),(nb,len(x)))
        bs=x[idx].mean(1); return x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
    summ={}
    for c in conds: summ[f"svis_{c}"]=boot(d[f"svis_{c}"].values)
    for a,b in [("C2","C1"),("C3","C1"),("C0m","C1"),("C2","C3")]:
        diff=(d[f"svis_{a}"]-d[f"svis_{b}"]).values
        summ[f"{a}-{b}"]=boot(diff)
    print(f"\n=== SUMMARY (mean, 95pct CI) N={len(d)} ===")
    for k,(m,lo,hi) in summ.items():
        star=" *SIG*" if (k.count("-")==1 and (lo>0 or hi<0)) else ""
        print(f"  {k:10s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}", flush=True)
    json.dump({k:list(v) for k,v in summ.items()}, open(os.path.join(args.out,"summary.json"),"w"), indent=1)
    print("\nsaved to", args.out)

if __name__=="__main__":
    main()
