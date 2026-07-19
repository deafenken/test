# Innovation plan + honest SOTA reconciliation (from frontier study `frontier_study.json`)

## SOTA landscape (verified from arXiv:2605.15864 PDF/HTML)
The only training-free single-pass SOTA on VS-Bench re-examination is the phenomenon paper's own **global-2×
attention amplification**: Qwen3-VL-8B-Thinking **36.6 → 54.8 (+18.2pp)** (their Table 9). Their **multi-turn user
instruction** is the practical ceiling: **36.6 → 67.5 (+30.9pp)** (Table 4). No other method (PAI/CAST/VisFlow/DMAS,
or the RL method VRRL 2607.02490) is evaluated on this benchmark. VRRL is a *training* method on grounding/navigation
(not VQA accuracy) — orthogonal to us.

## HONEST correction (three gates the frontier study exposed)
1. **Our earlier "beats global-2× (+8pp)" is PREMATURE / not valid.** Their global-2× really gets +18.2pp on their
   protocol; our reimplementation showed +0.000 because our operation (per-forward multiply-renormalize + a
   logit-preference metric on an N≈136 MathVerse subset) is **not** their setup (all-heads uniform 2× *during
   generation of R_b*, LLM-judged accuracy, full VS-Bench). We must **reproduce +18.2pp faithfully before any claim.**
2. **Metric mismatch:** our logit(A_b)>logit(A_a) proxy ≠ their post-swap **accuracy**. Switch to the real accuracy
   protocol (generate R_b, judge answer vs A_b) on the 3 VS-Bench scaffolds (Standard/Probe/Multi-turn).
3. **Partial scoop:** VisualSwap App. H / Table 15 already decomposes the user-turn recovery and shows bare
   turn-structure markers are negligible — so our C0m/marker decomposition is partly pre-empted. Novelty must move
   to the mechanism (magnitude-vs-provenance) and a NEW method.

## The genuinely-effective, non-scooped innovation to build
**PACD — Provenance-Contrastive Decoding (verdict: PURSUE, novelty 0.68, not scooped).** Single-pass: run two
branches over IDENTICAL image I_b and IDENTICAL re-look content, differing ONLY in role tokens — z_user (re-look as a
user turn) vs z_self (assistant continuation) — and decode from `softmax((1+λ)·z_user − λ·z_self)`. Target: **beat
single-pass global-2× (54.8 on 8B-Thinking)**; realistic +2–5pp over plain provenance-injection. It is training-free,
single-pass, and operationalizes "provenance is the lever" as a decoding rule.

## The decisive mechanistic result (publishable whichever way it lands)
**Accuracy-vs-Svis "money plot" with a magnitude-matched control.** Reproduce global-2× (+18.2pp) AND retune our
per-head amplifier so image-token Svis exactly equals global-2×'s, as PURE magnitude. Prediction: at matched Svis,
provenance recovers accuracy while pure magnitude does much less — showing the SOTA magnitude fix is suboptimal
because magnitude is not the causal lever (provenance is). This co-opts/explains the SOTA rather than just beating it.

## Full experiment grid (for a credible submission)
- **Benchmarks:** VS-Bench = 4 sources × 200 (MathVista, MathVerse, MathVision, MMMU-Pro). Dev on MathVerse; full
  4-source for the final table + generalization panel.
- **Metrics (all three per cell):** (1) faithful post-swap **accuracy** (their protocol); (2) Svis (post-softmax
  image attention) — for the money plot (accuracy up WITHOUT Svis up); (3) non-swap **retention** (over-correction check).
- **Methods:** self-floor · multi-turn-user ceiling · **global-2× (renorm + no-renorm)** · PAI · CAST · VisFlow · DMAS ·
  provenance-injection (ours) · **PACD (ours)** · magnitude-matched control (ours).
- **Models:** Qwen3-VL-8B Instruct+Thinking (primary), 32B-Thinking, ERNIE-4.5-VL-28B-A3B, Kimi-VL-A3B (generalization;
  VisualSwap publishes NO fix for ERNIE/Kimi — a genuinely open win). Seeds ≥3.

## Immediate next steps
1. Implement the **accuracy** metric (generate R_b, exact-match/normalized vs A_b) + the 3 scaffolds.
2. Reproduce **global-2× during generation** faithfully; verify ≈+18.2pp on 8B-Thinking MathVerse.
3. Build **PACD**; run the money plot (magnitude-matched control).
4. Scale to 4 sources + models + seeds.

Status: our earlier N≈136 logit results stand only as a *pilot signal*, not a SOTA claim. This plan replaces them.
