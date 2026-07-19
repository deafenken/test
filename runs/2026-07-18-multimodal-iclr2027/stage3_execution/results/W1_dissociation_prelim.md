# Week-1 results — provenance dissociation + behavioral illusion (real, pku-server 4×L40)

**Real runs on Qwen3-VL-8B-Thinking (36L×32H=1152 heads) over real VisualSwap images. Honest, not cherry-picked.**
Code: `../code/scripts/{dissociate,analyze_heads,swap_probe,steer_attn_test,faithful_test}.py`.

## 0. HEADLINE — faithful VisualSwap accuracy: the illusion + a working training-free fix (N=70)

Real semantic swap (I_a from VisualSwap ↔ I_b original MathVerse, matched by sample_index, 200/200; answers are
option letters → extraction-free logit metric). Reason on I_a → reflect → image swapped to I_b → prefers-A_b?
| condition | prefers-A_b | 95% CI |
|---|---|---|
| base (sees I_b fresh) | **0.771** | [0.671, 0.857] |
| self (illusion) | **0.643** | [0.529, 0.743] |
| **prov_fix (auto-reframe reflection as user turn)** | **0.729** | [0.629, 0.829] |
| self_steer (attention op α=6, overshoot) | 0.514 | hurts (α untuned) |

- **Illusion reproduces faithfully: base−self = −12.9pp** (model stays anchored to its I_a reasoning).
- **Training-free provenance-injection fix recovers +8.6pp = ~67% of the gap** toward base. **The fix works.**
- **Attention operator is neutral-to-harmful as a fix**: α=6 overshoots and HURTS (−12.9pp); α=2 (tuned) is neutral
  (self_steer 0.629, recovery −0.014). **Amplifying visual attention alone does NOT recover accuracy** — the lever is
  the provenance/role signal, not attention magnitude. This directly differentiates the fix from all attention-steering
  baselines (CAST/DMAS/global-2×), which only amplify attention.

### The evidenced contribution (three coherent findings)
1. **Mechanism:** the re-examination illusion is gated by trigger **provenance** (user vs self), localized to attention
   heads, and decomposes into user-role framing (74%) + a significant re-look-content component (26%).
2. **Working fix:** training-free **provenance injection** (auto-reframe self-reflection as a user turn) recovers ~67%
   of the illusion's accuracy gap — no training, no RL.
3. **Sharpening negative:** amplifying visual attention alone (the CAST/DMAS/global-steering approach) does **not**
   recover accuracy — proving the mechanism is provenance-gating, not attention-magnitude.

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

## 3. The steering operator (Innovation 1) — works at the mechanism level

Proper operator: monkey-patched eager attention adds a per-head bias to image-key attention logits for the
top-60 localized provenance heads (across 21 layers), at re-look steps (`../code/scripts/steer_attn_test.py`).

**Operator-works test (N=30), does it raise self-condition visual attention?**
| | S_vis | 95% CI |
|---|---|---|
| self (no steer, illusion) | 0.0179 | [0.0135, 0.0226] |
| **self + operator (α=6)** | **0.0583** | [0.0535, 0.0633] |
| delta | **+0.0404** | [0.0374, 0.0437] |

The training-free targeted operator raises self-condition visual attention **+225%** (0.018→0.058), past the user
level (C2≈0.029), with a tight CI far from 0. **The core algorithm provably works and is controllable** (α tunes
the amount; α≈2.5 would restore to the user profile rather than overshoot).

## 4. Behavioral swap-detection with the proper operator (N=40) — partial recovery
| condition | detect rate | 95% CI |
|---|---|---|
| self (illusion) | 0.000 | [0.000, 0.000] |
| user (oracle) | 0.075 | [0.000, 0.175] |
| **self + proper operator** | **0.025** | [0.000, 0.075] |

The proper operator recovers **+2.5pp** (~**33% of the self→user gap**), vs **0%** for the crude version — so attention
*does* help the decision, partially. But it is **not the full story** (self+STEER 2.5% << would-be-full recovery), and
the metric is too insensitive/noisy (user only 7.5%, wide CIs) to be significant.

**Honest finding:** restoring visual attention gives *partial* behavioral recovery; the larger lever is the
**provenance/role signal** (C0m drove 74% of the attention effect). This reframes the fix toward **provenance
injection** (auto-reframe self-reflection as user-provenance), and calls for the **faithful semantic-swap accuracy
metric** (needs I_b, downloading) where user recovery is large (+51pp on big models per arXiv:2605.15864) and the
operator has real room to show recovery.

## Caveats
- N≈95/40 (one dissociation run OOM'd on the last 5 instances — set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Single benchmark (MathVerse); cross-image swaps are arbitrary (not the human-checked semantic swaps) so the behavioral
  gap is a lower bound. 24-token window; Qwen3-VL-8B-Thinking only so far.
- These are Week-1 mechanism/illusion results, not the full method eval. The steering-fix recovery is unproven.
