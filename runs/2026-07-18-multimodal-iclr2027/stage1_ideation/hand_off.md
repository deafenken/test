# Hand-off: stage 1 (ideation) → stage 2 (method)

## What was decided

- **Chosen direction (C1-repaired)**: A **role-gated, training-free fix for the visual re-examination illusion in VLMs.** Full spec in `chosen.json`.
- The 3 original candidates all failed adversarial refutation (workflow `wf_73561b51-160`, 16 agents, 145 tool calls): C2 & C3 non-reparably scooped (arxiv:2605.14054, arxiv:2604.10219); C1's latent-depth premise falsified (arxiv:2605.12163). C1 was repaired via a substrate pivot onto the verified self-reflection-illusion phenomenon (arxiv:2605.15864).
- **Core hypothesis (falsifiable)**: the visual-attention surge behind genuine re-examination is gated by trigger-token **role/provenance** (user vs self-generated), localized to a **small set of attention heads**, and restorable **training-free** to recover the self-vs-user accuracy gap on swap-sensitive instances without degrading swap-invariant ones.
- **Primary claim**: a per-instance-gated, provenance-restoring intervention on those heads recovers most of the self-vs-user gap (arxiv:2605.15864) toward the user-prompt oracle, WITHOUT the collateral loss global steering (CAST arxiv:2605.04641, DMAS arxiv:2602.21704) incurs, and without RL.

## Two predictions Stage 2 must turn into testable hypotheses

1. **H1 (mechanism)**: In a 3-condition matched-CONTENT design (self-generated mid-decode / user-turn injection / assistant-role turn-boundary injection), per-head visual-attention mass responds to *role*, not just position — and the responsive heads form a small, causally-necessary set (ablating them removes the user-turn recovery).
2. **H2 (intervention)**: Restoring the user-role attention profile on those heads at self-reflection steps, gated by g(x), lifts VisualSwap detection + downstream accuracy on swap-sensitive instances toward the user-prompt oracle, with net-positive JOINT accuracy vs global steering (recover swap-sensitive AND preserve swap-invariant).

## Why this fits ICLR 2027 now

Mechanism-first (not a benchmark), on a phenomenon the field has only *diagnosed* (2605.15864, May 2026) and only *fixed via RL* (VRRL 2607.02490, Jul 2026). A training-free, causal, per-instance mechanistic account + fix is the open gap. Fully open-model, reproducible, 9-page-scopable.

## Constraints Stage 2 MUST respect (don't-touch list)

- **No RL.** The entire value prop vs VRRL (2607.02490) is training-free + mechanistic. RL also collides with the unproven GRPO-on-PCIe-4xL40 risk.
- **Training-free intervention + at most a tiny external probe** for the gate g(x). No full fine-tuning of the VLM.
- **The week-1 kill gate is mandatory and comes FIRST** (see `chosen.json::kill_conditions_week1`). The cheapest decisive test (VisualSwap internal-trigger validation, inference-only) runs before any building.
- **Collateral-free JOINT metric** (swap-sensitive recovery AND swap-invariant preservation) is the headline comparison vs global steering — not raw efficiency.
- Every `\cite` must be from `literature_pool.json`; `chosen.json::citations_to_verify_before_use` (CAST 2605.04641, DMAS 2602.21704, Look-Back 2507.03019, RVLM 2603.24224, VEPA 2606.17678) MUST be arXiv-verified before they enter method.md/plan.

## Open questions for Stage 2

- Does the role effect localize to few heads (LLaVA-1.5 fixed-token first for clean attribution)? Confirmed only by experiment.
- Exact operationalization of "user-role reference profile" restoration (attention re-scaling vs KV/role spoof vs positional spoof).
- Dataset triage: confirm VS-Bench availability/access; construct swap-sensitive vs swap-invariant slices for POPE/GQA/etc.
- Whether the external user-turn oracle is reachable internally, or only partially (bounds the headline).

## Files the next stage must read first

- stage1_ideation/chosen.json
- stage1_ideation/literature_pool.json
- stage0_setup/venue_profile.yaml
- run.yaml
- stage1_ideation/verification_workflow_result.json  (full adversarial vetting + scoop evidence)
