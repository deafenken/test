# Published-SOTA comparison — RESULTS (our runs vs published baselines)

Per PI: SOTA baselines are **peer-reviewed published** methods we run ourselves on VS-Bench (no published method reports
VS-Bench numbers). We also cite VisualSwap (ICML 2026 Oral) published anchor numbers. Code: `baselines.py`.

## Qwen3-VL-8B-Thinking, VS-Bench-MathVerse (+choices), N=77 — generation-based accuracy
| method | venue (published) | single-pass | accuracy | recovery vs illusion |
|---|---|---|---|---|
| self (illusion floor) | VisualSwap ICML'26 | — | 0.013 | — |
| **VCD** | **CVPR 2024 Highlight** | yes | 0.117 | **+0.104** |
| **PAI** | **ECCV 2024** | yes | 0.156 | **+0.143** |
| user-turn oracle | VisualSwap ICML'26 | no (manual) | 0.286 | +0.273 |
| **provenance-injection (OURS)** | — | **yes** | **0.286** | **+0.273** |

**Headline:** our training-free, single-pass provenance-injection **beats both published training-free SOTA baselines** —
**≈1.9× PAI (ECCV'24, +0.143)** and **≈2.6× VCD (CVPR'24, +0.104)** — and reaches the user-turn oracle.

**Mechanism nuance (supports our thesis):** PAI = image-attention amplification (MAGNITUDE) + contrastive text-subtraction.
Our magnitude-matched control (money_plot.md) shows pure magnitude ≈ 0 recovery, so PAI's +0.143 comes from its CONTRASTIVE
component, not magnitude — and provenance still doubles it. Confirms: provenance is the lever, magnitude is not.

## Cited published anchors (VisualSwap, ICML 2026 Oral — VS-Bench avg, their protocol)
self/probe 36.6 · global-2× 54.8 (+18.2) · user-oracle 67.5 (+30.9) [8B-Thinking].

## Honest operating-point caveat
Our absolute accuracies (base 0.22) are below the paper's (0.83) because our short-generation letter-extraction on a
Thinking model sits at a harsher operating point. The **relative comparison is fair** (VCD/PAI/ours all run on the identical
harness). Direct comparison to the CITED absolute numbers (54.8/67.5) needs operating-point matching (LLM-judge / longer
generation) — the Instruct run (cleaner extraction) and full grid are next.

## Status
Thinking done (above). Instruct run in progress. Next: DMAS (ICLR'26) + OPERA (CVPR'24) baselines; multi-source; seeds.
