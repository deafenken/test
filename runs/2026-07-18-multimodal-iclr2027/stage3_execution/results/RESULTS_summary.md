# Consolidated results — honest summary (Qwen3-VL-8B-Thinking, VisualSwap-MathVerse)

The one-screen honest state of the research. Details in the per-experiment files; raw data on pku-server.

## What is SOLID (real, reproduced across runs)
1. **The re-examination illusion is real and large.** Generation-based accuracy: base 0.60 → self **0.012** (the model
   almost never answers correctly after a self-reflection swap). Logit-metric earlier: base 0.69 → self 0.52.
2. **Provenance is the lever, magnitude is NOT — the decisive "money plot"** (`money_plot.md`, N=85):
   - provenance re-framing (prov_fix) recovers **+52.9 pp** and **exactly matches the manual user-turn oracle** (both 0.541).
   - attention amplification — global-2× **+2.4 pp**, magnitude-matched control **+4.7 pp** — barely moves accuracy.
   - **>10× gap.** The magnitude-matched control makes this causal: at matched visual-attention magnitude, provenance
     recovers and pure magnitude does not.
3. **Mechanism decomposition** (`W1_dissociation_prelim.md`): the visual-attention surge is gated by user-role
   provenance (localized to attention heads), with a significant re-look-content component; not turn-boundary alone.
4. **The steering operator works at the attention level** (+225% Svis) but is accuracy-neutral — consistent with (2).

## What is HONEST-NEGATIVE / NOT yet a SOTA claim
- **PACD (the fancier contrastive-decoding innovation) does NOT beat the simple provenance-injection** (λ=1.0: 0.459 <
  prov_fix 0.541; λ-sweep running). The simple auto-reframe is the effective method; PACD is so far a negative.
- **We have NOT reproduced the paper's global-2× +18.2 pp** (we get +2.4 pp) — our short-forced-letter accuracy is at a
  harsher operating point (self≈1% vs their probe≈29.5%). So **no direct "beats their 54.8" leaderboard claim is valid yet.**
- Single benchmark (MathVerse), single model, N≈85, single seed. Part of the marker decomposition is pre-empted by
  VisualSwap's appendix (Table 15).

## The defensible contribution (honest framing)
- **Scientific:** the VisualSwap re-examination illusion is a **provenance/role-gating** failure, not an
  attention-magnitude failure — demonstrated by the magnitude-matched money plot (the SOTA's own global-2× fix is
  suboptimal because it cranks the wrong variable).
- **Method:** a **training-free automatic provenance-injection** that matches the manual user-turn oracle (+52.9 pp).
- This co-opts/explains the SOTA rather than (yet) beating it on their leaderboard.

## Required to turn this into a submission (the remaining full-scale work)
1. **Protocol-matched reproduction** (full R_b generation + judged accuracy on Standard/Probe/Multi-turn scaffolds) to
   reproduce their base/probe/global-2× numbers exactly, then place prov_fix/PACD/mag-match on the same axis.
2. **Full grid:** 4 VS-Bench sources × models {8B-I/T, 32B-T, ERNIE-28B, Kimi} × ≥3 seeds × {accuracy, Svis, retention}.
3. **PACD λ-sweep** to completion (likely negative); if negative, drop PACD and lead with the money plot + auto-provenance.
4. Cross-model generalization panel (VisualSwap publishes NO fix for ERNIE/Kimi — a genuinely open result).

Version: pushed to github.com/deafenken/test.
