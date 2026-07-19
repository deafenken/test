# provlook — Provenance-Gated Visual Re-Engagement (training-free)

Stage-3 execution repo for run `2026-07-18-multimodal-iclr2027` (ICLR 2027).
Implements the locked `../../../stage2_method/{method.md, experiment_plan.yaml, pseudocode.py}`.
**Training-free**: inference + eager-attention capture + activation steering + a tiny supervised gate probe. No RL, no fine-tuning.

## Status
- **BLOCKED on GPU**: pku-server (4×L40) unreachable as of 2026-07-19 (jump-box SSH down, remote-side). This scaffold is CPU-authorable; runs start when the server returns.
- Primary data **secured**: `ChufanSHI/VisualSwap` (HF, public; 4 configs × 200 = 800).

## Layout (maps to compute phases P1–P6)
```
provlook/
  data.py        # VisualSwap + source-benchmark loaders; swap-sensitive/invariant slicing; localization/eval splits
  hooks.py       # eager-attention per-head capture; image-key logit rescaling operator (§4.1)
  dissociate.py  # P1: C1/C2/C3/C0m/C4 + PS matched-content conditions; C2-C3 / C3-C1 / C0m tests; MDE
  localize.py    # P2: provenance-differential ranking, FDR, split-clean nested-CV head selection
  gate.py        # P3/P4: per-instance abstain probe; swapped/OOD AUROC; margin-only baseline
  baselines.py   # P5: global-2x, CAST/DMAS (reimpl+tuned), PAI/TLVS/VCD; matched-budget strength tuning
  evaluate.py    # P6: final grid, Rec-vs-C2, joint seed+instance bootstrap, Pareto plots
  metrics.py     # Rec (C2 ceiling), collateral, non-inferiority, bootstrap/MDE utilities
configs/         # per-model YAML (model rev, dtype, eager attn, POSITION_GRID=7, subsample sizes)
scripts/         # entrypoints per phase; each logs git commit, seed, config, gpu, lib versions
tests/           # CPU smoke tests on a tiny VLM (hook correctness, ratio-preservation, split disjointness)
```

## Upstream
Built on VisualSwap (VLMEvalKit fork): `run_inference.py` applies the swap protocol, `download_data.py` fetches VS-Bench.
Our steering wraps its model forward; we do NOT alter its swap/eval logic (fair reuse).

## Week-1 GO/NO-GO (run first)
`scripts/p1_dissociation.py` — reproduce self/C2/C4 gaps + C2/C3/C0m/PS dissociation on Qwen3-VL-8B-Thinking,
measure eager-attention latency, MDE, and global-2× collateral (F3). GO iff gap≥15pp AND C2−C3 CI excludes 0
AND C3−C1 small AND C0m does not reproduce the surge. See `../../../stage2_method/experiment_plan.yaml::schedule_weeks[W1]`.
