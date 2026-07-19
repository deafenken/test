# Published-SOTA comparison plan (PI requirement: SOTA must be PEER-REVIEWED published, not arXiv preprints)

Verified via workflow (venues cross-checked against official proceedings / OpenReview; raw: `published_sota_verification.json`).

## Publication status (verified)
**PUBLISHED (usable as SOTA / anchor):**
- **VisualSwap "Are VLMs Seeing or Just Saying?"** — **ICML 2026 Oral** (arXiv:2605.15864 Comments = "ICML 2026 Oral").
  The problem+benchmark anchor. Its **global-2× fix (54.8 on 8B-Thinking, +18.2pp)** and **user-turn oracle (67.5, +30.9pp)**
  are PUBLISHED reference numbers we can cite directly.
- **DMAS** — Dynamic Multimodal Activation Steering — **ICLR 2026** (OpenReview accept; arXiv:2602.21704). Strongest recent
  published training-free mitigation → primary head-to-head baseline.
- **VCD** — Visual Contrastive Decoding — **CVPR 2024 Highlight** (arXiv:2311.16922).
- **PAI** — Paying More Attention to Image — **ECCV 2024** (arXiv:2407.21771). **THE contrast baseline**: its thesis is that
  amplifying image-attention MAGNITUDE reduces hallucination — exactly what our magnitude-matched control shows is insufficient.
- **OPERA** — Over-Trust Penalty + Retrospection — **CVPR 2024 Highlight** (arXiv:2311.17911).
- **HALC** (ICML 2024), **M3ID** (CVPR 2024), **ICD** (ACL 2024 Findings), **SID** (ICLR 2025) — optional additional published baselines.

**EXCLUDED as SOTA (arXiv preprint only):** VRRL (2607.02490, RL, preprint), CAST (2605.04641, preprint), VisFlow (2506.12609, preprint).

**Published benchmarks:** VS-Bench (ICML'26 Oral, our anchor), POPE (EMNLP'23), HallusionBench (CVPR'24), CHAIR (EMNLP'18),
MMHal-Bench (ACL'24 Findings), MMVP (CVPR'24). Auxiliary-only (benchmark preprints): MME, AMBER.

## The plan (all-published comparison)
Target benchmark = **VS-Bench (ICML'26 Oral, published)**. Because it is newly introduced, **no published mitigation method
reports VS-Bench numbers** → we RUN every baseline ourselves on VS-Bench under the swap protocol:
1. **Ours:** provenance-injection (single-pass, training-free) + magnitude-matched control.
2. **Published baselines (we run):** DMAS (ICLR'26), VCD (CVPR'24), PAI (ECCV'24), OPERA (CVPR'24) [+ optional HALC/M3ID/SID].
3. **Published anchors (we cite from VisualSwap ICML'26 Oral):** illusion lower bound (self re-look) + user-turn oracle upper
   bound + their global-2× number — all published reference points.

**Headline to establish:** provenance-injection recovers illusion accuracy toward the user-turn oracle, while the published
decoding/attention baselines (DMAS/VCD/PAI/OPERA) do NOT close the gap — and PAI (the published magnitude approach) is the
clean contrast confirming magnitude ≠ lever.

## Comparison table (to fill with our runs; SOTA rows are published)
| method | venue (published) | single-pass | 8B-Thinking VS-Bench acc | notes |
|---|---|---|---|---|
| self / illusion (VisualSwap anchor) | ICML'26 Oral | — | 36.6 (cite) | lower bound |
| global-2× (VisualSwap fix) | ICML'26 Oral | yes | 54.8 (cite) | their single-pass fix |
| user-turn oracle (VisualSwap) | ICML'26 Oral | no (manual) | 67.5 (cite) | upper bound |
| DMAS | ICLR 2026 | yes | [we run] | strongest published mitigation |
| VCD | CVPR 2024 | yes | [we run] | contrastive decoding |
| PAI | ECCV 2024 | yes | [we run] | magnitude approach (contrast) |
| OPERA | CVPR 2024 | yes | [we run] | over-trust penalty |
| **provenance-injection (OURS)** | — | **yes** | **[we run — target ≈ 67.5]** | auto-oracle, training-free |

## Status
Compute BLOCKED (pku-server relay down) + local drive read-only (committing via GitHub API). Need to run: DMAS/VCD/PAI/OPERA
+ ours on VS-Bench (choices-fixed harness ready). Implementing the 4 published baselines is the main remaining engineering.
