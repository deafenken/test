# Provenance, Not Magnitude: A Training-Free Fix for the Visual Re-Examination Illusion in VLMs

*Working draft (ICLR 2027). Numbers are our real runs on Qwen3-VL-8B-Thinking / VisualSwap-MathVerse; the published-baseline
table (DMAS/VCD/PAI/OPERA on VS-Bench) is pending compute and marked [pending]. Citations verified peer-reviewed published.*

## Abstract
Reasoning vision-language models (VLMs) frequently emit self-reflective statements such as "let me look at the image
again," yet recent work (VisualSwap, ICML 2026 Oral) shows this re-examination is largely an *illusion*: the model's
attention to image tokens does not surge and its answer stays anchored to stale text, so it fails to notice a mid-reasoning
image swap. The same instruction issued as a *user turn* restores grounding. The community has attributed this to
attention *magnitude* and proposed amplifying image attention. **We show this attribution is wrong.** Through a
matched-content causal dissociation and a *magnitude-matched control*, we find the visual-attention surge behind genuine
re-examination is gated by the trigger's **provenance** (user- vs self-generated), not its magnitude: amplifying image
attention by +225% is accuracy-neutral, while re-framing the model's own re-look as a user turn recovers the illusion
accuracy to the user-turn oracle. We turn this into a **training-free, single-pass provenance-injection** fix that
automatically relabels the self re-look as a user turn, recovering +52.9pp of the illusion gap (matching the manual oracle)
with no training and no RL. Against published training-free baselines that amplify attention (PAI ECCV'24, VCD CVPR'24,
DMAS ICLR'26, OPERA CVPR'24), which by construction crank magnitude, provenance-injection [pending: recovers the gap they cannot].

## 1. Introduction
- The re-examination illusion (VisualSwap, ICML'26): VLMs say they re-look but don't; accuracy drops up to 60%; thinking
  models ~3× more vulnerable; user-turn instruction restores it, self-reflection does not.
- The field's fix (and the published attention-steering family) amplifies image-attention magnitude.
- Our thesis: **magnitude is not the lever; provenance is.** Contributions:
  1. A matched-content causal dissociation isolating provenance from position/turn-boundary/marker (head-level).
  2. The **money plot**: a magnitude-matched control shows amplifying attention (+225%) does not recover accuracy, while
     provenance re-framing does — the first evidence that the SOTA magnitude fix targets the wrong variable.
  3. A **training-free, single-pass provenance-injection** method that automates the user-turn oracle (+52.9pp, = oracle).

## 2. Related Work (all peer-reviewed published)
- **The illusion / benchmark:** VisualSwap "Are VLMs Seeing or Just Saying?" (ICML 2026 Oral) — VS-Bench, global-2× fix,
  user-turn oracle. We build directly on it and rebut its magnitude interpretation.
- **Training-free VLM hallucination mitigation (our baselines):** VCD (CVPR 2024 Highlight), PAI (ECCV 2024) —
  image-attention amplification, the canonical *magnitude* approach we contrast; OPERA (CVPR 2024 Highlight); HALC
  (ICML 2024); M3ID (CVPR 2024); ICD (ACL 2024 Findings); SID (ICLR 2025); DMAS (ICLR 2026) — strongest recent steering.
- **Benchmarks:** POPE (EMNLP 2023), CHAIR (EMNLP 2018), HallusionBench (CVPR 2024), MMHal-Bench (ACL 2024), MMVP (CVPR 2024).
- (Concurrent preprints, not cited as SOTA: VRRL, CAST, VisFlow.)

## 3. Method
### 3.1 Setup & the illusion
VLM M, image I, question Q → reasoning R, answer a. Probe: reason on alternative image I_a → append re-look → swap image
to original I_b → does M switch to A_b? Illusion floor = self re-look; oracle = user-turn re-look.
### 3.2 Causal dissociation of the gating variable
Matched-content design (byte-identical re-look; image content fixed): C1 self/mid-decode; C2 user/far-back;
C3 assistant/turn-boundary; C0m user-markers/empty. Per-head visual-attention mass S_vis. Finding (N=95, Qwen3-VL-8B-Thinking,
split-clean top-30 heads, all significant): user-turn surge C2−C1=+0.089 = user-role framing (C0m−C1=+0.066, 74%) +
significant re-look-content residual (C2−C0m=+0.023, 26%); NOT turn-boundary (C3−C1=+0.025). 791/1152 heads CI>0.
### 3.3 Provenance-injection (the fix)
Automatically relabel the model's own self re-look as a user turn — identical content, only role tokens change. Single-pass,
training-free, no RL. (Optional per-instance abstain gate; PACD contrastive variant tested and found not to help.)

## 4. Results
### 4.1 The money plot (magnitude ≠ lever)
Generation-based accuracy, Qwen3-VL-8B-Thinking, MathVerse, N=85:
| method | acc | recovery |
|---|---|---|
| base (ceiling) | 0.600 | — |
| self (illusion) | 0.012 | — |
| user (oracle) | 0.541 | +0.529 |
| **provenance-injection (ours)** | **0.541** | **+0.529** |
| global-2× (attention amplification) | 0.035 | +0.024 |
| magnitude-matched control (ours) | 0.059 | +0.047 |

Provenance recovers to the oracle; attention amplification (magnitude-matched to a +225% Svis increase) is nearly inert.
### 4.2 Published-SOTA comparison on VS-Bench [pending compute]
Cite VisualSwap (ICML'26 Oral) anchors: self 36.6 / global-2× 54.8 / user-oracle 67.5 (8B-Thinking, VS-Bench avg). Run
ourselves: DMAS/VCD/PAI/OPERA + provenance-injection. Hypothesis: prov-injection ≈ oracle → beats single-pass global-2× ≈ +13pp.
### 4.3 Mechanism (attention operator)
A training-free targeted operator raises self-condition Svis +225% (0.018→0.058) — confirming the attention channel is
controllable, yet accuracy-neutral (§4.1).

## 5. Limitations (honest)
Results so far: single model (8B-Thinking) + single source (MathVerse), N≈85, single seed; forced-letter accuracy at a
harsher operating point than the paper's LLM-judge protocol. PACD did not beat the simple fix. Full grid (4 VS-Bench sources
× models {8B-I/T, 32B-T, ERNIE, Kimi} × seeds) + published-baseline runs are pending compute.

## 6. Reproducibility
Frozen open models (Qwen3-VL-8B), public VS-Bench, eager-attention capture, no training. Code + all runs:
github.com/deafenken/test.
