# Domain frame (Stage 1, written 2026-07-18)

- **Core area**: multimodal learning, vision-language (user-fixed). Sub-areas in scope: VLM reasoning/hallucination
  mechanisms, efficient visual tokenization/fusion, multimodal representation alignment, test-time/decoding-side methods.
- **Implicit compute constraints**: 4×L40-48GB (no NVLink), 600 GPU-h, 9 weeks → NO pretraining from scratch, no >13B
  training; feasible = full FT ≤3B, LoRA ≤13B (LLaVA/Qwen-VL class), inference-, probing-, and decoding-heavy studies.
- **Venue-preferred shape (ICLR)**: mechanism/insight-first with ablation-backed explanation; falsifiable hypothesis;
  open models/datasets; tight 9-page scoping. Red flags: benchmark sweeps, closed-API dependence, +0.3-metric deltas.
- **Strategy**: hunt for gaps where a *causal/mechanistic* claim about VLMs can be tested cheaply at ≤13B scale and
  turned into a small, principled method — the "insight + method + ablations" ICLR trifecta.
