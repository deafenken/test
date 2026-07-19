# Hand-off: stage 2 (method) → stage 3 (execution)

## What is being implemented and why

A **training-free** study of the VLM *visual re-examination illusion* (`arxiv:2605.15864`): (Part A) a matched-content causal dissociation of **what gates the visual-attention surge** — token role/provenance vs absolute position/recency vs turn-boundary — with head-level localization; (Part B) a **targeted, per-instance-gated attention-rescaling fix** that restores the "engaged" profile on the identified heads, aiming to recover the self→user accuracy gap on swap-sensitive instances **without** the collateral that global steering incurs. No RL, no fine-tuning — inference + eager-attention capture + activation steering + a tiny supervised gate probe. Full spec: `method.md`; locked plan: `experiment_plan.yaml`.

## Headline metric

**Recovery fraction** `Rec = (acc_method − acc_self)/(acc_oracle − acc_self)` on **swap-sensitive** VisualSwap instances, reported jointly with **collateral Δ** on swap-invariant + POPE + MME. The thesis is "recover **without** collateral" — both are headline; a recovery win with collateral ≥ global steering is NOT a success.

## The Week-1 GO/NO-GO (run FIRST, before building anything)

On Qwen3-VL-8B-Thinking, ≥200 swap-sensitive instances, eager attention, byte-identical re-exam string, image content held fixed:
**GO iff** (i) self-vs-oracle swap-detection gap ≥ 15pp AND (ii) the **role main effect** survives position+boundary controls (C2−C1 bootstrap CI excludes 0, C3≈C1). If NO-GO → Fallback A (reframe to the winning variable: turn-boundary or recency gating) or Fallback B (mechanistic-characterization paper). Both are pre-committed and ICLR-viable — see `experiment_plan.yaml::fallbacks`.

## Do NOT silently change (locked)

- **Regime = training-free.** No RL, no VLM fine-tuning (the entire value prop vs VRRL `2607.02490`).
- **Hypotheses / success (S1–S5) / failure (F1–F5) criteria** are pre-registered; check failure criteria and escalate, do not paper-wash.
- **Primary metric = Recovery fraction; collateral is co-headline.**
- **Datasets/baselines** as listed; VS-Bench is primary. CAST/DMAS must be reimplemented faithfully (no public code) with POPE home-turf sanity before cross-eval.
- **Models**: Qwen3-VL-8B-Thinking primary; nothing >48GB required (30B-A3B FP8 is a stretch only).

## Stage-3 must read first

- stage2_method/method.md
- stage2_method/experiment_plan.yaml
- stage2_method/pseudocode.py
- stage2_method/stress_test.md
- stage1_ideation/chosen.json
- **BLOCKER**: pku-server (4×L40) unreachable as of 2026-07-19 (jump-box SSH down, remote-side). Stage 3 cannot start until connectivity is restored. Week-0 tasks that need NO GPU (secure VisualSwap data / build harness scaffolding locally) can proceed.

## Week-0 (no-GPU) prep that can start now

1. Secure VisualSwap data (HF request / email authors) or stand up the public-source reconstruction.
2. Read TLVS (`2606.07647`) and VisFlow (`2506.12609`) in full to confirm the per-instance-gate novelty before locking.
3. Scaffold the eager-attention capture + head-rescaling harness (CPU-testable on a tiny model).
