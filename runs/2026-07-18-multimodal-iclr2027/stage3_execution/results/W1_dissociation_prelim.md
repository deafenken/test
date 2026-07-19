# Week-1 preliminary results — provenance dissociation (real, on pku-server 4×L40)

**Status: preliminary (small N, short measurement window). Honest, not cherry-picked.**
Model class validated; the numbers below are the first real runs, to be scaled up.

## Setup
- Models: Qwen2.5-VL-7B-Instruct (28L×28H=784 heads); **Qwen3-VL-8B-Thinking** (36L×32H=1152 heads, faithful model class of arXiv:2605.15864).
- Data: real VisualSwap images (I_a), MathVerse_MINI, **N=20** instances.
- Measure: per-head visual-attention mass S_vis over the first **8** re-look continuation tokens, eager attention.
- Conditions (byte-identical reflection string P, image content fixed):
  C1 self/mid-decode · C2 user-turn · C3 assistant-boundary · C0m user-markers+empty-content.
- Code: `../code/scripts/{dissociate,analyze_heads}.py`.

## Aggregate S_vis (mean over ALL heads) — Qwen3-VL-8B-Thinking
| contrast | mean | 95% CI | sig |
|---|---|---|---|
| C2−C1 (provenance) | −0.0015 | [−0.0054, +0.0014] | no |
| C2−C3 (user vs assistant boundary) | +0.0002 | [−0.0032, +0.0027] | no |
| C3−C1 (boundary) | −0.0016 | [−0.0064, +0.0029] | no |
| C0m−C1 (marker/sink) | −0.0020 | [−0.0058, +0.0004] | no |

**Aggregate is null** — expected, since H1/H2 predict a head-*concentrated* effect that the all-head mean washes out.

## Per-head localization — Qwen3-VL-8B-Thinking (the real test)
- **C2−C1:** per-head max **+0.1025** at head (L12,h18); top-20 mean +0.0379; **363/1152 heads have 95% CI > 0**.
  Top provenance heads: L12h18, L3h21, L14h7, L14h6, L12h4, L6h16, L3h11, L1h11, L22h17, L13h17.
- **C2−C3:** max +0.073 (L14h7); **496/1152 heads CI>0** — a broad user-vs-assistant-boundary differential.
- Concentration of positive C2−C1 mass: top-30 heads = 27%, top-50 = 37%, top-100 = 55%. Moderately concentrated (not a tiny 5-head set).

## The confound the design caught (as pre-registered)
- **C0m (user markers, empty content) also activates the same top heads** (368/1152 CI>0, same top head L12h18). So part of the C2 effect is **turn-marker / sink-token insertion**, not pure re-look provenance. The role-vs-marker-vs-boundary dissociation is not yet clean — exactly the F2 risk we pre-registered.

## Reading
There **is** a head-localized, trigger-conditioned visual-routing response (Innovation 1's substrate exists). Its exact gating variable (provenance vs marker vs boundary) is **not yet resolved** at N=20 / 8-token window. Next: (i) scale N and the measurement window, (ii) faithful thinking-model illusion setup, (iii) implement the swap-based **accuracy** gap so the fix can be shown to *recover accuracy*, (iv) proper localization/eval split. Pre-committed fallbacks (A: boundary/recency gating; B: mechanistic characterization) remain in force if provenance is falsified.

## Qwen2.5-VL-7B-Instruct (pilot, weaker illusion) — for reference
Aggregate final (N=20): C1=0.0234, C2=0.0270, C3=0.0242, C0m=0.0233 — C2>C1 in direction, noisy.
