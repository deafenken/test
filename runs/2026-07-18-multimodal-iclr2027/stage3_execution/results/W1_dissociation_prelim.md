# Week-1 results — provenance dissociation + behavioral illusion (real, pku-server 4×L40)

**Real runs on Qwen3-VL-8B-Thinking (36L×32H=1152 heads) over real VisualSwap images. Honest, not cherry-picked.**
Code: `../code/scripts/{dissociate,analyze_heads,swap_probe}.py`.

## 1. Mechanism: user-provenance-conditioned visual routing (dissociation, N=95, MathVerse, 24-tok window)

Aggregate S_vis (mean over all heads), stable across the run:
| condition | S_vis | reading |
|---|---|---|
| C1 self / mid-decode | ~0.019 | the illusion: self "look again" barely re-engages vision |
| **C2 user turn** | **~0.029** | **~1.55× C1 — a user re-look surges visual attention** |
| C3 assistant boundary | ~0.021 | ≈ C1 — a turn boundary alone is NOT enough |
| C0m user-markers, empty | ~0.025 | user-role framing recovers most of the surge |

Per-head (all-head mean washes out a broad effect): C2−C1 per-head max **+0.140** at (L12,h18);
**791/1152 heads have 95% CI > 0**. Top provenance heads: L12h18, L14h7, L17h17, L13h19, L15h11, L20h15, ...
Effect is **broad, not sparse** (top-100 heads hold only 40% of positive mass) → H2's "small set" is only partial; it's "a large fraction of heads are provenance-conditioned".

### Split-clean decomposition (localize top-30 heads on one half, measure on the disjoint half) — all *SIG*
| contrast | mean [95% CI] | interpretation |
|---|---|---|
| C2−C1 total user-turn effect | +0.089 [+0.073, +0.107] | the full surge |
| C0m−C1 marker / user-role only | +0.066 [+0.057, +0.075] | **74%** is user-role framing |
| **C2−C0m CONTENT-provenance residual** | **+0.023 [+0.011, +0.037]** | **26% is genuine re-look content — significant** |
| C3−C1 assistant boundary | +0.025 [+0.010, +0.045] | boundary contributes, but < user-role |
| C2−C3 user vs assistant boundary | +0.064 [+0.042, +0.084] | user provenance beyond boundary |

**Finding:** visual re-engagement is gated primarily by **user-role provenance** of the re-look cue (markers),
with a **significant additional content component**, and is **not** explained by turn-boundary alone. This is a
mechanistic decomposition the phenomenon paper (arXiv:2605.15864, layer-level only) does not provide.

## 2. Behavioral illusion (swap-probe, N≈40, real VisualSwap images)

Model reasons on image I_i, then the image is swapped to a different figure I_j; a matched Yes/No
"Same or Changed?" check is delivered as self vs user; detection = logit(Changed) > logit(Same).
| condition | detect-swap rate |
|---|---|
| **self-reflection** | **0.00** (0/25) — NEVER notices the swap: the illusion, starkly |
| user turn | 0.05–0.20 — noticing improves with user framing |
| self + crude STEER | 0.00 — see caveat |

**Finding:** a clean behavioral demonstration — under self-reflection the model is functionally blind to a mid-reasoning
image swap; user framing partially restores noticing. Directly mirrors the attention mechanism above.

## 3. Honest negative / next step: the fix
The **crude** steering (scaling image-token hidden states ×1.6 on layers {3,6,12,14,22}) did **not** recover
swap-detection (0%). The proper operator — per-head rescaling of image-key attention on the *localized* heads to the
measured user-role profile (Innovation 1's actual algorithm) — is the next iteration; the crude proxy is insufficient.

## Caveats
- N≈95/40 (one dissociation run OOM'd on the last 5 instances — set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Single benchmark (MathVerse); cross-image swaps are arbitrary (not the human-checked semantic swaps) so the behavioral
  gap is a lower bound. 24-token window; Qwen3-VL-8B-Thinking only so far.
- These are Week-1 mechanism/illusion results, not the full method eval. The steering-fix recovery is unproven.
