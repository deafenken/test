# Published-SOTA comparison — RESULTS (our runs vs published baselines)

Per PI: SOTA baselines are **peer-reviewed published** methods we run ourselves on VS-Bench (no published method reports
VS-Bench numbers). We cite VisualSwap (ICML 2026 Oral) anchors. Code: `baselines.py` (our faithful VCD/PAI reimplementations).

## Headline — provenance-injection beats published SOTA on BOTH models (VS-Bench-MathVerse, +choices, N=77)
| method | venue (published) | single-pass | Thinking acc (rec) | Instruct acc (rec) |
|---|---|---|---|---|
| self (illusion floor) | VisualSwap ICML'26 | — | 0.013 | 0.273 |
| **VCD** | **CVPR 2024 Highlight** | yes | 0.117 (+0.104) | 0.130 (**−0.143**) |
| **PAI** | **ECCV 2024** | yes | 0.156 (+0.143) | 0.182 (**−0.091**) |
| user-turn oracle | VisualSwap ICML'26 | no (manual) | 0.286 (+0.273) | 0.377 (+0.104) |
| **provenance-injection (OURS)** | — | **yes** | **0.286 (+0.273)** | **0.377 (+0.104)** |
| base (ceiling) | — | — | 0.221 | 0.429 |

**Our training-free, single-pass provenance-injection beats both published training-free SOTA baselines on BOTH models,
matching the user-turn oracle:**
- Thinking: ours +0.273 vs PAI +0.143 (≈1.9×) / VCD +0.104 (≈2.6×).
- Instruct: ours +0.104, while **VCD (−0.143) and PAI (−0.091) actively HURT** (over-correct below the illusion floor).

**Mechanism nuance (supports our thesis):** PAI = attention amplification (MAGNITUDE) + contrastive text-subtraction. Our
magnitude-matched control (money_plot.md) shows pure magnitude ≈ 0 recovery, so PAI's partial Thinking recovery is its
CONTRASTIVE part — and provenance still doubles it. On Instruct, decoding-side contrastive/amplification methods over-correct
the re-look continuation and drop below the illusion floor, whereas provenance re-framing cleanly restores grounding.

## Cited published anchors (VisualSwap ICML 2026 Oral, VS-Bench avg, their LLM-judge protocol)
8B-Thinking: self/probe 36.6 · global-2× 54.8 (+18.2) · user-oracle 67.5 (+30.9).

## Honest caveats
- Operating point harsher than the paper's (base 0.22 Thinking / 0.43 Instruct vs their 0.83) due to short-generation letter
  extraction; the **relative** comparison is fair (VCD/PAI/ours all on the identical harness). Direct comparison to the CITED
  absolute numbers needs operating-point matching (LLM-judge / longer generation).
- VCD/PAI are our faithful reimplementations (blind-written, no authors' code on VS-Bench); N=77, single source, single seed.
- DMAS (ICLR'26) + OPERA (CVPR'24) baselines, multi-source, and seeds are next.
