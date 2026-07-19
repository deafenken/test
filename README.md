# Provenance-Gated Visual Re-Engagement in VLMs

Research repo for an ICLR 2027 submission (mechanism-first, training-free) on the
**visual re-examination illusion** in vision-language models: reasoning VLMs say
"let me look again" but their visual attention does not surge and their answers stay
text-anchored (Shi et al. 2026, VisualSwap, arXiv:2605.15864). We ask **what gates the
visual-attention surge behind genuine re-engagement** — trigger *provenance* (user vs
self-generated), *position/recency*, or *turn-boundary* — and build a training-free fix.

Full pipeline state lives under `runs/2026-07-18-multimodal-iclr2027/` (venue setup →
ideation → method → execution), managed by the `auto-research` skill.

## Three algorithm/mechanism innovations (the contribution) — now EVIDENCED

Real runs on Qwen3-VL-8B-Thinking over real VisualSwap images (see `runs/.../stage3_execution/results/`).

1. **Provenance-gated visual re-engagement (mechanism).** The re-examination illusion is gated by the *provenance*
   (user- vs self-generated) of the re-look trigger, localized to attention heads. A user re-look surges visual
   attention ~1.55× vs self; split-clean head decomposition (all significant): user-role framing 74% + **significant
   re-look-content residual +0.023 (26%)**, not explained by turn-boundary. Not a small sparse set — 791/1152 heads.

2. **Training-free provenance-injection fix (algorithm).** Auto-reframe a self-generated re-look as a user turn (no
   human, no training, no RL). On the faithful VisualSwap accuracy metric it **recovers +8.6 pp (~67%) of the −12.9 pp
   illusion gap** (base 0.771 → self 0.643 → fix 0.729).

3. **Attention-magnitude is NOT the lever (sharpening mechanism result + differentiation).** A tuned training-free
   per-head attention-rescaling operator raises self-condition visual attention +225% yet is **neutral-to-harmful on
   accuracy** — proving the illusion is provenance-gating, not attention-magnitude, and directly differentiating from
   all attention-steering baselines (CAST/DMAS/VisFlow/global-2×) which only amplify attention.

Supporting methodological contributions: the matched-content causal **dissociation protocol** (self/user/boundary/
marker/position-sweep) and the extraction-free faithful **letter-logit accuracy** metric (I_a↔I_b matched 200/200).
Still open: the per-instance abstain gate; full N × multi-benchmark × seeds grid; tuned baselines.

## Status (2026-07-19)
- Stages 0–2 complete and adversarially stress-tested (see `runs/.../stage{0,1,2}_*`).
- Stage 3 executing on pku-server (4×L40): eager-attention harness validated; first
  provenance-dissociation runs on Qwen2.5-VL-7B and Qwen3-VL-8B-Thinking done.
- Compute/data notes in `runs/.../stage2_method/week0_notes.md`.

## Reproducibility
Frozen open models (Qwen2.5-VL-7B, Qwen3-VL-8B-Thinking), public VisualSwap data,
eager attention for per-head capture. No training of the VLM; no RL.
