"""
Per-head localization analysis (H1 sharper test + H2).
Loads perhead_*.npz (each: C1,C2,C3,C0m arrays [L,H] = per-head visual mass for one instance).
Computes, per head, the mean provenance differential across instances and tests whether a
SMALL set of heads carries a strong, consistent effect — the aggregate mean washes this out.
"""
import glob, os, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv)>1 else "/home/intern/provlook_out_q3think"
conds = ["C1","C2","C3","C0m"]
files = sorted(glob.glob(os.path.join(OUT,"perhead_*.npz")))
assert files, f"no npz in {OUT}"
stacks = {c: [] for c in conds}
for f in files:
    d = np.load(f)
    for c in conds: stacks[c].append(d[c])
for c in conds: stacks[c] = np.stack(stacks[c])   # [N, L, H]
N, L, H = stacks["C1"].shape
print(f"N={N} instances, L={L} layers, H={H} heads/layer, total heads={L*H}")

# per-head mean contrasts across instances
def contrast(a,b):
    diff = stacks[a] - stacks[b]         # [N,L,H]
    m = diff.mean(0)                     # [L,H] mean over instances
    # paired bootstrap CI per head (light)
    rng = np.random.RandomState(0)
    idx = rng.randint(0, N, (500, N))
    bs = diff.reshape(N,-1)[idx].mean(1).reshape(500,L,H)
    lo, hi = np.percentile(bs,2.5,0), np.percentile(bs,97.5,0)
    return m, lo, hi

for a,b in [("C2","C1"),("C2","C3"),("C3","C1"),("C0m","C1")]:
    m,lo,hi = contrast(a,b)
    flat = m.flatten()
    agg = flat.mean()
    # top provenance-differential heads
    order = np.argsort(flat)[::-1]
    topk = 20
    top = order[:topk]
    sig_pos = ((lo.flatten()>0)).sum()   # heads with CI>0
    print(f"\n### {a}-{b}: aggregate mean over all heads = {agg:+.5f}")
    print(f"    per-head max = {flat.max():+.5f} at (L{top[0]//H},h{top[0]%H}); "
          f"top-{topk} mean = {flat[top].mean():+.5f}; heads with 95pctCI>0: {sig_pos}/{L*H}")
    if a=="C2" and b=="C1":
        print("    top-10 provenance heads (layer,head): "
              + ", ".join(f"(L{i//H},h{i%H}:{flat[i]:+.4f})" for i in order[:10]))
        np.save(os.path.join(OUT,"prov_diff_C2C1.npy"), m)   # for later steering target

# how concentrated? cumulative share of positive differential in top-K heads
m,_,_ = contrast("C2","C1")
pos = np.clip(m.flatten(),0,None); pos_sorted = np.sort(pos)[::-1]
tot = pos_sorted.sum()
for K in [5,10,30,50,100]:
    print(f"    top-{K} heads hold {100*pos_sorted[:K].sum()/max(tot,1e-9):.1f}% of total positive C2-C1 mass")

# ---- MECHANISM DECOMPOSITION at the top provenance heads (split-clean) ----
# Localize top-K heads on a LOCALIZATION half, then decompose the effect on the EVAL half.
# Tests: is the C2>C1 surge PURE PROVENANCE (C2>C0m, content-sourced) or MARKER/BOUNDARY
# re-anchoring (C0m ~= C2, and C3 also up)?
print("\n### MECHANISM DECOMPOSITION at top-30 heads (split-clean localize/eval) ###")
half = N//2
loc = {c: stacks[c][:half] for c in conds}
ev  = {c: stacks[c][half:] for c in conds}
prov_loc = (loc["C2"]-loc["C1"]).mean(0).flatten()          # localize on first half
topH = np.argsort(prov_loc)[::-1][:30]                       # top-30 provenance heads (LOC split)
def ev_contrast(a,b):
    d=(ev[a]-ev[b]).reshape(ev[a].shape[0],-1)[:,topH]       # [N/2, 30] at localized heads, EVAL split
    x=d.mean(1)                                              # per-instance mean over the 30 heads
    rng=np.random.RandomState(1); idx=rng.randint(0,len(x),(2000,len(x)))
    bs=x[idx].mean(1); return x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
for lab,(a,b) in [("C2-C1 (total user-turn effect)",("C2","C1")),
                  ("C0m-C1 (marker/sink only)",("C0m","C1")),
                  ("C2-C0m (CONTENT provenance residual)",("C2","C0m")),
                  ("C3-C1 (assistant boundary)",("C3","C1")),
                  ("C2-C3 (user vs assistant boundary)",("C2","C3"))]:
    mm,lo,hi=ev_contrast(a,b); sig="*SIG*" if (lo>0 or hi<0) else ""
    print(f"    {lab:38s} {mm:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}")
print("    Reading: C2-C0m>0 & sig => genuine CONTENT-provenance; else marker/boundary re-anchoring (Fallback A).")
