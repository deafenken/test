# Method stress test (Phase 5) — results and resolutions

Adversarial stress test run as workflow `wf_e08066bc-ca9` (4 hostile ICLR reviewers reading the on-disk artifacts). Raw: `stress_test_result.json`. As-locked prediction was **3.5–4.5 (reject)**; with all must-fixes + favorable Week-1 → **5.5–6.5 (borderline-accept)**. All 7 blockers were pre-registration/framing defects — fixed here with **zero GPU**. This file records the 9 canonical stress questions and how the (revised) `method.md`/`experiment_plan.yaml` answer them.

## Q1 — "Is this just method X under a different name?"
Resolved in method.md §7 (de-gerrymandered table with per-cell phrases). The single load-bearing novel atom is **provenance-conditioned (self-vs-user), per-instance-abstaining restoration of visual-routing heads at re-look steps, applied to the re-examination illusion**. Verified distinct from: VRRL (RL, no mechanism), phenomenon-paper (global/diagnosis), CAST/DMAS (always-on, caption/context-conditioned, no abstain), **VisFlow** (token-TYPE system/visual/text, not user-vs-self provenance — read in full W0), **TLVS** (per-TOKEN strength, no abstain, calibration-trained — read in full W0), PAI (global amplification), Gaze Heads (no provenance dim, no gate).

## Q2 — "Does the math predict the empirical claim?"
Reviewer found two gaps, both fixed: (a) the primary metric now scores recovery against the **C2 ceiling the operator actually targets** (not the recency-inclusive C4), with the recency residual reported separately (§5) — resolving the S1↔F2 contradiction; (b) §4.1 no longer claims to "reproduce the engaged profile" — it "moves each head's marginal image-mass toward the C2 target," concedes the cross-layer composition footprint, uses a **per-instance** target (not a population-mean clamp), and **measures downstream divergence from true C2** as a reported quantity.

## Q3 — "Simplest alternative explanation for the gain?"
Two: **image-recency** (C4 re-inserts the image recently; IKOD `2508.03469`, debiasing `2508.17807`) and **turn-boundary/sink-token** insertion (a user turn is a boundary with marker tokens). Controls added: **C2** (user, far-back, no re-insert) isolates provenance from recency; **C0m** (user markers, empty U) isolates sink-token insertion; **C3** (assistant boundary) isolates boundary; **PS** (position sweep) traces recency. The role test is the boundary-matched **C2−C3**, not the confounded C2−C1.

## Q4 — "How does it fail?" (and the fallbacks)
Pre-committed two-tier fallback (§8, plan `fallbacks`): **A** — role falsified but a training-free trigger works → reframe to the winning variable (boundary/sink-reset or recency KV re-prefill), still gated, still beats global-2×. **B** (minimum guaranteed contribution) — the first head-level matched-content dissociation of what drives the surge, strictly deeper than the phenomenon paper's layer-level analysis; committed as ICLR-sufficient independent of whether role/fix survive.

## Q5 — "Cheapest experiment that refutes the central claim?"
The **Week-1 GO/NO-GO** (≤84 GPU-h): self-vs-C2 gap ≥15pp AND C2−C3 CI excludes 0 AND C3−C1 small AND C0m does not reproduce the surge. Cheapest decisive test of the whole thesis; also reports MDE and the F3 global-2× collateral precondition.

## Q6 — "Is the benchmark choice biased?"
VisualSwap is math-centric (4 STEM benchmarks). Mitigations: **held-out swapped split** for deployment-AUROC, a **non-math OOD** gate-eval set (anti-shortcut), and **N-augmentation** via public-source reconstruction if MDE fails. Collateral measured on POPE+MME (non-math, subsampled uniformly). Gate labels on unswapped source are flagged as a **re-look-benefit proxy**, with AUROC re-reported on the true swapped distribution.

## Q7 — "Infinite compute — still matters?"
Yes: an **absolute** accuracy fix on a latent capability the model fails to deploy, not a compute trade-off. The gap is not sampling luck; more decode budget does not close it.

## Q8 — "Reviewer 2's most damning comment?" + response
> "Strip the fallbacks and the headline is two coin-flips: that 'role' survives dissociation from the boundary tokens it is physically identical to, and that global-2× is actually harmful."

**Response:** Both are now *pre-registered Week-1 measurements with pre-committed reframes*, not assumptions: provenance is tested at matched boundary (C2−C3) with a marker-only control (C0m); global-2× collateral (F3) is measured in W1 **before** building the apparatus, with a committed reframe if <2pp. The paper's **guaranteed** contribution (Fallback B: the first head-level matched-content dissociation) clears the bar even if both coin-flips land badly.

## Q9 — "Is this secretly an evaluation paper?"
No. The contribution is a **mechanism** (what gates visual re-engagement) + a **training-free intervention** (provenance-gated head restoration). Remove the tables and the causal dissociation + the method remain. The two evaluation/design contributions (the matched-content dissociation protocol, the collateral-on-swap-invariant metric) are listed **separately** from method properties in §7, not padded into the novelty conjunction.

## Residual risks (experiment-only, cannot be edited away — see plan `failure_criteria`)
F1 scale-down (<15pp gap at 8B) · role genuinely falsified (→ Fallback A, territory near IKOD/boundary work) · novel atom eroded by VisFlow/TLVS once fully re-read (checked W0: survives) · no collateral headroom (F3) · population-clamp/independent-solve ceiling · S2 CI separation may not resolve at achievable N (→ non-inferiority-only). Most-probable stacked-risk landing = Fallback B (honest, ICLR-viable, borderline). This is disclosed, not hidden.
