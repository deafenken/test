# Week-0 prep log (no-GPU tasks)

## ✅ VisualSwap / VS-Bench data — SECURED (2026-07-19)
- HF dataset `ChufanSHI/VisualSwap` is **public, non-gated**, parquet.
- Configs: `MathVista_MINI` (200), `MathVerse_MINI` (200), `MathVision` (200), `MMMU_Pro_10c_COT` (200) = **800**, matching arxiv:2605.15864.
- Fields include `image`, `question`, `answer`, and (MMMU config) `original_answer`.
- Repo tree: 4 config dirs + README + .gitattributes. Last modified 2026-05-15.
- The plan's swap-pair reconstruction fallback is therefore **not needed**.
- Note: confirm the exact swap-pairing convention (which record holds I vs I_b) from the README during Stage-3 W1.

## ⏳ pku-server (4×L40) — STILL UNREACHABLE
- Jump-box SSH (port 22) times out; remote-side outage. Retried repeatedly 2026-07-18/19.
- Blocks all GPU phases. My local Tailscale is healthy. Needs lab-side fix.

## ✅ Nearest-competitor reads — DONE (2026-07-19), novel atom SURVIVES
- **TLVS (arxiv:2606.07647)**: per-TOKEN adaptive *strength*, always-on (no abstain), NO provenance conditioning, "requires minimal training for calibration" (not fully training-free), general hallucination. → does NOT do per-instance abstain or provenance or re-examination.
- **VisFlow / Dual-Level Attention Intervention (arxiv:2506.12609)**: conditions on token-TYPE (system/visual/text), NOT user-vs-self provenance; always-on; targets first-generation hallucination, not re-examination; training-free. → distinct from our self-vs-user provenance + abstain gate.
- **2508.17807 (Attention Debiasing for Token Pruning)**: recency-bias + attention-sink debiasing for pruning → SUPPORTING evidence for our recency/sink confound, not a scoop.
- Conclusion: load-bearing novel atom (self-vs-user provenance + per-instance abstain + re-examination illusion) is unoccupied. §7 table updated with precise per-cell phrasing.
