# Hand-off: stage 0 → stage 1 (ideation)

## What was decided

- Target: **ICLR 2027 main conference**, planning deadline **2026-09-24** (abstract 2026-09-19); dates are
  aggregator-sourced and must be re-verified when iclr.cc/2027 goes live. ~9 weeks of runway from today.
- Domain (user-chosen): **multimodal learning (vision-language)**. The gap within it is Stage 1's job.
- Compute: **4×L40 48GB** on pku-server (no NVLink assumed). Planning gate 600 GPU-hours.
  Feasibility ceiling: full fine-tune ≤3B params, LoRA/adapters ≤13B, inference-/analysis-heavy studies cheap.
  **Server currently unreachable** (jump-box tailscale DERP down) — does not block ideation, blocks Stage 3.
- Venue assets: official ICLR 2026 CFP/AuthorGuide/template used as interim baseline (2027 unpublished);
  logged as fallback in `latex_source.json`.

## What ICLR rewards (shape ideation accordingly)

- Mechanism/insight-first contributions with ablation-backed explanations — NOT pure benchmark/eval papers
  (orchestrator hard constraint #8 also forbids defaulting to eval papers).
- Falsifiable claims validated with strong baselines, multiple seeds, tight scoping that fits 9 pages.
- Reproducible setups (open models/datasets; avoid closed-API-dependent pipelines).

## Risky/unconvincing patterns for this venue

- Incremental leaderboard deltas without insight; small-scale results overclaimed; closed-source dependence;
  "we evaluated N models on M benchmarks" framing.

## Open questions for the next stage

- Which multimodal sub-area offers a defensible, compute-feasible gap (e.g. VLM hallucination mechanisms,
  efficient multimodal fusion/tokenization, multimodal representation alignment, test-time multimodal reasoning)?
- What baselines are reproducible on 4×L40 within budget?

## Don't-touch list

- Target venue/deadline (user-set); domain = multimodal (user-set); 9-page format; integrity rules 1–8.

## Files the next stage must read first

- run.yaml
- stage0_setup/venue_profile.yaml
- stage0_setup/cfp.md
