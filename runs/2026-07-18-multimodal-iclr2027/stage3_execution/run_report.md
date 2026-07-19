# Stage 3 — Run report (Week-1 execution)

Hardware: pku-server, 4×L40-46GB (shared), env `nlpcc_t2` (transformers 4.57.3, torch 2.4+cu121, eager attention).
Models: Qwen2.5-VL-7B-Instruct (cached), **Qwen3-VL-8B-Thinking** (modelscope). Data: VisualSwap (MathVerse) + original
MathVerse (I_b) matched 200/200 by sample_index. Full results: `results/W1_dissociation_prelim.md`.

## Headline number
On the faithful VisualSwap accuracy metric (Qwen3-VL-8B-Thinking, N=70): the re-examination illusion reproduces
(**base 0.771 → self 0.643, −12.9pp**), and a **training-free provenance-injection fix recovers +8.6pp (~67% of the gap)**.

## Executed as planned
- Built + validated an eager-attention per-head capture harness (smoke test passed).
- Matched-content provenance dissociation (C1/C2/C3/C0m) on real VisualSwap images, N=95 — the Week-1 mechanism gate.
- Split-clean per-head localization + mechanism decomposition (localize/eval halves).
- Implemented the training-free per-head attention-rescaling operator (monkey-patched eager attention).
- Reproduced the faithful swap protocol (I_a→I_b) with an extraction-free letter-logit accuracy metric.

## Key findings
1. **Mechanism (H1):** user-turn re-look surges visual attention ~1.55× vs self (C2 0.029 vs C1 0.019, aggregate);
   split-clean top-30-head decomposition all significant — user-role markers 74%, **content-provenance residual
   +0.023 (26%, SIG)**, turn-boundary smaller. 791/1152 heads CI>0 but effect is **broad, not sparse** → H2's
   "small set" only partially holds.
2. **Operator works (Innovation 1):** the training-free targeted operator raises self-condition visual attention
   +225% (0.018→0.058, tight CI) — provably functional and tunable.
3. **Faithful illusion + fix:** −12.9pp illusion; provenance-injection fix +8.6pp (~67% recovery).
4. **Nuance:** attention alone is necessary-but-not-sufficient behaviorally (α=6 operator overshoots visual attention
   and HURTS faithful accuracy: −12.9pp); the provenance/role signal is the effective lever. α-tuning in progress.

## Deviations from plan
- **Hardware:** used 4×L40 as designed, but access was only restored after diagnosing a DERP-relay SSH timeout
  (needs ConnectTimeout≥120). Original 30B-A3B stretch not attempted.
- **Model:** Qwen3-VL-8B-Thinking substituted from modelscope (HF blocked server-side); Qwen2.5-VL-7B used for the pilot.
- **Metric:** the pre-registered Recovery-fraction (vs C2 ceiling) was complemented by a faithful letter-logit
  accuracy metric on the real semantic swap (cleaner given MathVerse letter answers) — reported alongside, not instead.
- **Behavioral swap-probe** (arbitrary cross-image) proved insensitive (user only ~10%); superseded by the faithful test.
- Scope: one benchmark (MathVerse) so far; N=70–100 (not the full 5-seed grid); baselines (CAST/DMAS) not yet run.

## Failed runs
- Two dissociation runs OOM'd on the last ~5 instances (long sequences + attention capture) → fixed with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; analysis used the completed 95.
- Several large-parquet transfers over the DERP relay truncated/corrupted → resolved by downsizing images (26MB→4.4MB).
- Crude hidden-state steering did not recover behavior (0%) → replaced by the proper attention operator.

## Compute consumed
~a few L40-hours total (inference/probing only; no training). Well under the 600 GPU-h budget; the study is cheap
because it is training-free.

## Next
- α-sweep for the attention operator (find the restore-not-overshoot point).
- Full N=200 × multiple benchmarks (MathVista/MMMU) × seeds; add tuned CAST/DMAS/global-2× baselines; the abstain gate.
