# SOTA comparison — their published numbers (cited) vs our method (our runs)

**Approach (per PI directive):** SOTA numbers are taken **directly from the published paper** (not reimplemented);
we run **only our method** on the same protocol and slot our number beside theirs. Validity requirement: our
Base/Probe must land at their operating point — enforced by the choices-fixed harness.

## Cited SOTA — VisualSwap "Are VLMs Seeing or Just Saying?" (arXiv:2605.15864, verified from PDF)
VS-Bench accuracy averaged over the 4 sources (Tables 4 & 9):

| model | Base | Probe (self illusion) | **global-2× (single-pass SOTA)** | multi-turn USER (manual oracle) |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 69.1 | 46.6 | **54.5 (+7.9)** | 58.2 (+11.6) |
| Qwen3-VL-8B-Thinking | 76.0 | 36.6 | **54.8 (+18.2)** | 67.5 (+30.9) |
| Qwen3-VL-235B-Instruct | 81.1 | 61.3 | — | 77.9 (+16.6) |
| Qwen3-VL-235B-Thinking | 88.8 | 34.1 | — | 85.4 (+51.3) |

- **global-2× is the single-pass training-free SOTA to beat.** No fix reported for ERNIE-4.5-VL-28B / Kimi-VL-A3B.
- **multi-turn USER** is their *oracle* — needs a MANUAL 2nd user turn with new content; NOT single-pass.

## Our method vs the cited SOTA (the claim to test)
Our **provenance-injection** = automatically reframe the model's OWN self re-look as a user turn (no human, no new
content, only role relabeling) — a **single-pass, training-free** method that should reach the *user-turn oracle*
recovery automatically.

| method | single-pass? | training-free? | 8B-Thinking | 8B-Instruct |
|---|---|---|---|---|
| global-2× (their SOTA, cited) | yes | yes | 54.8 | 54.5 |
| multi-turn USER (their oracle, cited) | **no (manual)** | yes | 67.5 | 58.2 |
| **provenance-injection (OURS)** | **yes** | **yes** | **[pending — target ≈ 67.5]** | **[pending — target ≈ 58.2]** |

**Hypothesis:** prov_fix ≈ the user-turn oracle → **beats single-pass SOTA global-2× by ≈ +13 pp (Thinking) / +3.7 pp
(Instruct)** by AUTOMATING the oracle in one pass. Our money-plot (magnitude-matched control) explains WHY global-2×
underperforms: it cranks attention magnitude, not provenance.

## Validity gate (before filling our numbers)
Our Base/Probe must ≈ theirs on the same source(s). Earlier our Base was 0.26–0.33 (prompt omitted CHOICES) → FIXED:
`faithful_choices.parquet` uses VisualSwap `query_wo` (full query + choices + option-letter instruction). Pending:
Instruct+choices run (`vsc`) to confirm Base ≈ 0.69–0.83, then fill prov_fix. Full 4-source average for the final table.

## Status (2026-07-19 late)
Two external blockers: (1) pku-server relay unstable; (2) local /Volumes/ORICO drive read-only. Work safe on GitHub +
server. Datasets: MathVerse ready (choices); MathVista/MMMU-Pro originals downloaded/ing; MathVision pending.
