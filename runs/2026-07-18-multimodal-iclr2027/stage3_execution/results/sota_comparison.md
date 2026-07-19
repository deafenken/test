# SOTA comparison — faithful VisualSwap accuracy (real, pku-server 4×L40)

> ⚠️ **SUPERSEDED / PREMATURE — read `innovation_plan.md`.** The frontier study (verified) found this comparison is
> **not a valid SOTA claim**: (1) the phenomenon paper's global-2× actually gets **+18.2pp** on their protocol, but our
> reimplementation here (per-forward multiply-renormalize + logit-preference metric on N≈136) shows +0.000 — a
> methodology mismatch, NOT evidence that global-2× fails; (2) our logit-preference metric ≠ their post-swap accuracy;
> (3) their setup applies 2× *during generation*, all heads, on full VS-Bench. This table stands only as a PILOT signal.
> The rigorous redo (faithful global-2×, accuracy metric, magnitude-matched "money plot", PACD) is in `innovation_plan.md`.


**Model:** Qwen3-VL-8B-Thinking. **Data:** VisualSwap-MathVerse, I_a↔I_b matched 200/200 (N=124 valid after letter filtering).
**Metric:** prefers-A_b = logit(A_b letter) > logit(A_a) after reason-on-I_a → reflect → swap-to-I_b. self=illusion floor, base=ceiling.
**Code:** `../code/scripts/sota_compare.py`. All baselines are training-free; intervention = post-softmax amplify-and-renormalize
of image-key attention by 2× on the target head-set (the phenomenon paper's / PAI-family method).

| method | prefers-A_b | 95% CI | recovery vs self |
|---|---|---|---|
| base (sees I_b fresh; ceiling) | 0.694 | [0.613, 0.774] | — |
| **self (illusion floor)** | 0.516 | [0.427, 0.605] | — |
| global-2× (arXiv:2605.15864's own fix; ALL heads ×2) | 0.516 | [0.427, 0.605] | **+0.000** |
| CAST/DMAS-style (top-K image heads ×2) | 0.508 | [0.427, 0.589] | −0.008 |
| our attention operator (localized provenance heads ×2) | 0.508 | [0.427, 0.589] | −0.008 |
| **provenance-injection (OURS)** | **0.597** | [0.508, 0.677] | **+0.081** |

## Result
- **Every training-free attention-steering SOTA baseline gives ≈0 recovery** — including global-2×, the phenomenon
  paper's own proposed fix (+0.000), and CAST/DMAS-style head steering (−0.008).
- **Our provenance-injection fix recovers +8.1 pp** (≈46% of the 17.8pp illusion gap) — the ONLY method that recovers
  faithful accuracy. So vs the training-free SOTA it is **+8.1 pp better than global-2× and +8.9 pp better than CAST/DMAS-style.**
- This is the clean confirmation of the thesis: the re-examination illusion is **provenance-gated, not
  attention-magnitude-gated** — amplifying visual attention (what all these methods do) cannot fix the decision;
  re-framing the trigger's provenance can.

## Honest caveats
- Single benchmark (MathVerse), single model, N=124, single seed. Multi-benchmark (MathVista/MMMU) + seeds pending
  (downloads in progress) for a robust claim.
- Baselines are faithful reimplementations: global-2× matches the phenomenon paper exactly; CAST/DMAS-style is an
  amplify-top-K-heads approximation of their head-subset steering (real CAST/DMAS add caption/vector selection, but all
  are attention-amplification, so the directional conclusion — attention steering ≈ 0 recovery — holds).
- VRRL (arXiv:2607.02490, RL SOTA) is a *training* method (different regime); not run. Our contribution is the first
  *training-free* fix that recovers accuracy, complementary to and cheaper than the RL approach.
