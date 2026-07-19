# Provenance-Gated Visual Re-Engagement in VLMs

Research repo for an ICLR 2027 submission (mechanism-first, training-free) on the
**visual re-examination illusion** in vision-language models: reasoning VLMs say
"let me look again" but their visual attention does not surge and their answers stay
text-anchored (Shi et al. 2026, VisualSwap, arXiv:2605.15864). We ask **what gates the
visual-attention surge behind genuine re-engagement** — trigger *provenance* (user vs
self-generated), *position/recency*, or *turn-boundary* — and build a training-free fix.

Full pipeline state lives under `runs/2026-07-18-multimodal-iclr2027/` (venue setup →
ideation → method → execution), managed by the `auto-research` skill.

## Three algorithm/mechanism innovations (the contribution)

1. **Trigger-source-conditioned visual-routing heads + a training-free steering operator.**
   We identify a small, localizable set of attention heads whose image-token attention is
   gated by the *source* of the re-look trigger, and a training-free operator that restores
   the "engaged" attention profile on exactly those heads at re-look steps. (Mechanism +
   algorithm.) *Early evidence (Qwen3-VL-8B-Thinking): a concentrated head set — top head
   L12h18, C2−C1 = +0.10 — with a marker/boundary confound under active disentanglement.*

2. **Per-instance illusion-abstain gate.** A tiny supervised probe on internal features
   that decides *whether* an instance is illusion-bound and swap-sensitive, so the
   intervention is surgical and collateral-free — unlike always-on global steering
   (CAST/DMAS/VisFlow). (Algorithm/framework.)

3. **Cross-layer joint attention re-solve.** Steering heads that compose across layers by
   naive independent per-head clamping does not reach the target *joint* state; we solve the
   heads jointly / iteratively toward the measured engaged joint profile. (Algorithm.)

Supporting methodological contributions (not counted above): the matched-content causal
**dissociation protocol** (self / user / boundary / marker / position-sweep) that identifies
the gating variable, and the **collateral-on-swap-invariant** evaluation metric.

## Status (2026-07-19)
- Stages 0–2 complete and adversarially stress-tested (see `runs/.../stage{0,1,2}_*`).
- Stage 3 executing on pku-server (4×L40): eager-attention harness validated; first
  provenance-dissociation runs on Qwen2.5-VL-7B and Qwen3-VL-8B-Thinking done.
- Compute/data notes in `runs/.../stage2_method/week0_notes.md`.

## Reproducibility
Frozen open models (Qwen2.5-VL-7B, Qwen3-VL-8B-Thinking), public VisualSwap data,
eager attention for per-head capture. No training of the VLM; no RL.
