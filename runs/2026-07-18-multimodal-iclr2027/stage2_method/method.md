# Method — Disentangling What Gates Visual Re-Engagement in VLMs, and a Training-Free Targeted Fix

**Working title:** *Seeing or Just Saying? What Gates Visual Re-Engagement in VLM Self-Reflection, and a Training-Free Gated Fix*

**Run:** `2026-07-18-multimodal-iclr2027` · **Venue:** ICLR 2027 (main) · **Regime:** training-free (inference + attention/activation probing + activation steering + a tiny external gate probe). **No RL, no VLM fine-tuning.**

> Contract Stage 3 implements. Grounded in `verification_workflow_result.json` and revised against the adversarial stress test `stress_test_result.json` (all 7 must-fixes applied). Pre-registered companion: `experiment_plan.yaml`. Deviations are logged, not silent.

---

## 1. Problem setup and the phenomenon

A vision–language model $M$ maps an image $I$ and a question $Q$ to a reasoning trace $R$ and answer $a$. Reasoning VLMs emit **self-reflective spans** ("let me look at the image again"). Shi et al. (2026, `arxiv:2605.15864`), via the **VisualSwap** probe, show these are largely an *illusion of visual re-examination*: after $M$ reasons over $I$, the image is silently replaced by a similar-but-different $I_b$; $M$ overwhelmingly fails to notice, its visual-token attention does **not** surge at the re-look span, and its answer stays anchored to stale text context. The **same** instruction as a *user turn* does surge visual attention and restores accuracy. The field calls this *"a failure of control, not capability"* and offers the user-prompt as a **workaround** — with no account of **why** identical instruction content behaves differently by source, and no internal fix.

Our object is the **self-vs-user gap**, and the open question:

> **What variable gates the visual-attention surge that produces genuine re-engagement — the *provenance* of the trigger tokens (user-provided vs self-generated), the *absolute position/recency* of the image in the sequence, or *turn-boundary* re-anchoring (attention-sink reset)?**

Answering this is the primary scientific contribution; a training-free **targeted, gated** fix that follows is the secondary contribution. **We do not assume provenance is the answer** — the design is built to *dissociate* the three candidates, and every downstream claim is pre-committed to reframe onto whichever variable wins (§8).

### 1.1 Why this is not already solved (see §7 for the full table)

- **VRRL** (`arxiv:2607.02490`) fixes the symptom with **RL** (no attention analysis, no provenance, no gate, no swap-detection) — a disjoint solution family and a foil for a training-free mechanistic account.
- **Phenomenon paper** (`arxiv:2605.15864`) is diagnosis-first; its only fix amplifies visual attention **globally (2×, all heads, always-on)** — no localization, no gate, no collateral analysis.
- **CAST/DMAS** (`2605.04641`, `2602.21704`) steer head subsets but **always-on, every instance**, conditioned on caption-salience / context, **not on trigger provenance**, with **no whether-to-steer decision**.
- **VisFlow** (`arxiv:2506.12609`) conditions on token **type** (system/visual/text), **not** user-vs-self provenance; always-on; targets first-generation hallucination, not re-examination.
- **TLVS** (`arxiv:2606.07647`) does per-**token** steering **strength** (not a per-**instance** abstain decision), no provenance, needs calibration training, general hallucination.
- **Gaze Heads** (`arxiv:2606.14703`) localizes visual-routing heads but with **no provenance dimension**, **no gate**, and no re-examination application.

**The single load-bearing novel atom:** a *per-instance-**abstaining** intervention conditioned on the **self-generated-vs-user provenance** of the re-look trigger, applied to the visual re-examination illusion, and validated by a matched-content causal dissociation of provenance from position and turn-boundary.* No prior work occupies it. We defend **this atom** against the full neighbor set (§7), not a 6-way conjunction.

---

## 2. Notation

| Symbol | Role | Domain | First use |
|---|---|---|---|
| $M$ | the VLM under study (frozen) | — | §1 |
| $I,\ I_b$ | original image, semantically-swapped image | pixel space | §1 |
| $Q,\ R_a$ | question tokens; model reasoning (assistant role) | token seq | §1 |
| $U$ | re-examination instruction, **byte-identical** across conditions | token seq | §3.1 |
| $\rho$ | **provenance** of $U$: user-turn vs assistant/self-generated | $\{\mathrm{usr},\mathrm{slf}\}$ | §3.1 |
| $\beta$ | **turn-boundary** flag (is $U$ at a fresh turn boundary) | $\{0,1\}$ | §3.1 |
| $\pi$ | **image position**: original far-back slot vs recent re-inserted slot; also swept | token index | §3.1 |
| $\mathcal H$ | all attention heads $(\ell,h)$ | — | §3.2 |
| $S^{(\ell,h)}_t$ | **visual attention mass** $=\sum_{j\in\mathcal I_t}\alpha^{(\ell,h)}_{t,j}$ at re-look step $t$ | $[0,1]$ | §3.2 |
| $\delta^{(\ell,h)}$ | **provenance-differential** score of a head (§3.3), from the C2−C3 contrast | $\mathbb R$ | §3.3 |
| $\mathcal H_K$ | the $K$ selected provenance-conditioned visual-routing heads (chosen on a **disjoint** split) | $\subseteq\mathcal H$ | §3.3 |
| $\hat p^{(\ell,h)}(x)$ | **per-instance engaged target**: predicted C2 visual mass for head $(\ell,h)$ on instance $x$ | $[0,1]$ | §4.1 |
| $g(x),\ \tau$ | per-instance **abstain gate** (steer iff $g>\tau$) | $[0,1]$ | §4.2 |
| $\mathrm{acc}_{\mathrm{self/C2/C4/method}}$ | accuracies under conditions (§3.1) | — | §5 |

**Provenance** $\rho$ is the chat-template role segment $U$ occupies (user vs assistant markers + their KV context), independent of $U$'s literal string, which is byte-identical across conditions. Provenance ≠ VisFlow's token-*type* (system/visual/text) distinction.

---

## 3. Part A — Causal dissociation of the gating variable

### 3.1 The matched-content, boundary-matched design

We cross the three candidate gating variables while holding $U$ byte-identical and the image **content** fixed (the re-encoded $I_b$). Crucially, the primary role contrast is **boundary-matched** (both at $\beta=1$), because in a chat template user-provenance is *instantiated by* a turn boundary — so C2−C1 confounds provenance with boundary and is **only descriptive**.

| Cond. | $\rho$ | $\pi$ (image) | $\beta$ | isolates |
|---|---|---|---|---|
| **C1** | self | far-back | 0 | the illusion baseline (self-generated $U$ mid-decode) |
| **C2** | **user** | far-back | 1 | user provenance **without** image re-insertion |
| **C3** | self (assistant) | far-back | 1 | **turn-boundary** at matched (self) provenance |
| **C0m** | user *markers*, **empty/neutral $U$** | far-back | 1 | **marker/sink-token** insertion, no re-look content |
| **C4** | user | **recent (re-inserted)** | 1 | oracle upper bound (confounds provenance+recency) |
| **PS** | self | **swept** | 0 | **recency** curve (image relocated across a position grid) |

- **Primary role test:** the *provenance* effect is $(\mathrm{C2}-\mathrm{C3})$ — matched boundary, differing only in user-vs-self provenance.
- **Boundary control:** $(\mathrm{C3}-\mathrm{C1})$ must be small (boundary alone does not explain the surge).
- **Marker/sink control (C0m):** if user *markers with empty $U$* already reproduce the C2 surge, the effect is sink-token insertion, not provenance.
- **Recency control (PS + C4):** PS traces the recency curve (motivated by IKOD `arxiv:2508.03469`, attention-decay-with-length; and attention-debiasing `arxiv:2508.17807`); C4 is the recency-inclusive oracle.
- **Sequence-length/position matching:** total measurement-query length and absolute image→query token distance are padded/aligned equal across C1/C2/C3/C0m, so the position basis is not silently varied by user-wrapper tokens.

### 3.2 Measuring the surge

Per head $(\ell,h)$ and re-look step $t$, capture post-softmax attention (**eager** attention) and compute $S^{(\ell,h)}_t=\sum_{j\in\mathcal I_t}\alpha^{(\ell,h)}_{t,j}$, averaged over the span. Instance signal $S_{\text{vis}}$ follows the phenomenon paper's mean text→image attention aggregated over $\mathcal H_K$.

### 3.3 Isolating provenance and localizing heads (de-circularized)

**Identification caveat (stated, not assumed away).** There is no natural $(\rho=\mathrm{usr},\beta=0)$ cell in a deployed chat template, so the provenance×boundary interaction is **not point-identified**. We therefore report a *descriptive decomposition* under an explicit no-interaction caveat, and anchor the claim on the boundary-matched contrast C2−C3 plus the C0m marker control — **not** on an additive regression whose additivity we cannot test.

Per-head **provenance-differential** score $\delta^{(\ell,h)} = \mathbb E[S^{(\ell,h)}\!\mid\!\mathrm{C2}] - \mathbb E[S^{(\ell,h)}\!\mid\!\mathrm{C3}]$, estimated with instance-bootstrap CIs and **FDR (Benjamini–Hochberg) correction across all $\approx$1150 heads** (Qwen3-VL-8B: 36 layers × 32 heads). Head selection is **split-clean**: $\mathcal H_K$ and the knee-$K$ are chosen on a **localization split**; recovery, the random-head control, and S3 are reported on a **disjoint evaluation split** (nested CV over instances), with head-set stability across folds reported. The size-matched random-head control is drawn and scored on the same eval split.

> **Falsifiable H1 (role).** On the eval split: the aggregate provenance statistic on the pre-registered head set has an instance-bootstrap 95% CI excluding 0 for **C2−C3**, **C3−C1** is small (CI includes 0 or effect < ½ of C2−C3), **and** C0m does not reproduce the C2 surge. If C3 or C0m explains it → turn-boundary/sink mechanism (Fallback A). If PS drives it → recency mechanism (Fallback A).
> **Falsifiable H2 (localization).** A split-clean $K\le 50$ ($<\!15\%$ of heads) reaches $\ge80\%$ of full-set recovery on the eval split, CI-separated from the size-matched random control.

---

## 4. Part B — Training-free, targeted, per-instance-gated fix

### 4.1 The steering operator (honest scope)

At re-look steps only, for each $(\ell,h)\in\mathcal H_K$ we apply an additive shift to image-key pre-softmax logits, uniform over image keys and zero elsewhere:
$$\tilde z^{(\ell,h)}_{t,j}=z^{(\ell,h)}_{t,j}+c^{(\ell,h)}_t\,\mathbb 1[j\in\mathcal I_t],\qquad c^{(\ell,h)}_t:\ \textstyle\sum_{j\in\mathcal I_t}\mathrm{softmax}(\tilde z)_j=\text{target}.$$
This **preserves the within-image and within-text relative attention** (ratio-preservation is exact) and moves only the image-vs-text balance. **Honest scope (revised per stress test):** (i) the operator *moves each head's marginal image-mass toward* its target — it does **not** reproduce the joint C2 state, because $c^{(\ell,h)}$ is solved per head while heads **compose across layers** (steering layer $\ell$ shifts downstream heads' logits that were measured offline in the un-steered C2 context); we therefore **measure and report the downstream activation divergence from true C2**, not just softmax-local invariance. (ii) The target is **per-instance** $\hat p^{(\ell,h)}(x)$ (a small predictor conditioned on cached pre-trigger features), not a population-mean clamp — a constant clamp cannot let hard instances surge above the mean; the population-mean variant is retained only as an ablation (§6.5). We compare **independent** vs **joint/iterative re-solve** to the C2 state as an ablation. Frozen weights, no training.

### 4.2 The per-instance abstain gate

Global steering pays a **collateral** cost where the language prior was already correct. The gate spends the intervention only where it helps: $g(x)=\sigma(w^\top\varphi(x))$, a tiny probe on **cached** features $\varphi(x)$ (pre-trigger visual-mass on $\mathcal H_K$ vs target, and answer-logit margin). Steer iff $g(x)>\tau$ ($\tau$ calibrated to an intervention budget).

**Train/deploy honesty (revised).** Labels on a held-out *unswapped* source split (MathVista/MathVerse/MathVision/MMMU-Pro) are a **re-look-benefit proxy** (they contain no $I_b$), used to avoid learning 800-pair math shortcuts; but AUROC is **additionally reported on a held-out *swapped* VisualSwap split** — the true deployment distribution — and on a **non-math OOD** gate-eval set to test the anti-shortcut claim. We include an **answer-margin-only** baseline gate to show internal features add *illusion-specific* signal, and report the AUROC bootstrap CI. No RL — a supervised probe on frozen features.

> **Falsifiable H3 (fix).** On swap-**sensitive** instances, targeted+gated steering recovers $\ge$ the pre-registered fraction of the **role-attributable (C2)** gap (§5), with collateral $\ge -2$pp on swap-**invariant** + POPE + MME, **non-inferior** to no-op and with a between-method advantage over *tuned* global-2×/CAST/DMAS.
> **Falsifiable H4 (gate).** Gate AUROC on the held-out **swapped** split clears the pre-registered bar **and** the gated policy Pareto-dominates always-on-targeted in the (recovery, collateral) plane. (If AUROC is weak but gating still Pareto-dominates, the claim rests on the latter — F5.)

---

## 5. Metrics (ceiling-corrected)

The steering restores the **C2** (user-provenance, far-back) profile, which structurally cannot access C4's recency term. So the primary recovery is scored against the **role-attributable C2 ceiling**, and the residual recency is reported separately:
$$\mathrm{Rec}=\frac{\mathrm{acc}_{\text{method}}-\mathrm{acc}_{\text{self}}}{\mathrm{acc}_{\mathbf{C2}}-\mathrm{acc}_{\text{self}}},\qquad
\text{recency residual}=\mathrm{acc}_{\mathbf{C4}}-\mathrm{acc}_{\mathbf{C2}}\ \text{(reported, not targeted)}.$$

- **Headline reporting** is the set of **absolute** accuracies $\{\mathrm{acc}_{\text{self}},\mathrm{acc}_{\mathrm{C2}},\mathrm{acc}_{\mathrm{C4}},\mathrm{acc}_{\text{method}}\}$ with **paired instance-bootstrap CIs**; $\mathrm{Rec}$ is **secondary** (a normalized summary).
- **Domain:** $\mathrm{Rec}\in(-\infty,\infty)$ — targeted steering can **overshoot** the C2 ceiling, so $\mathrm{Rec}>1$ is possible and is reported as-is (not clipped). A **minimum-denominator floor** is pre-registered: if $\mathrm{acc}_{\mathrm{C2}}-\mathrm{acc}_{\text{self}}<15$pp (CI-adjusted), Rec is **suppressed** and only absolute accuracies are reported; on the secondary model, if the denominator is non-positive, Rec is undefined and only absolutes are used.
- **Secondary:** collateral $\Delta_{\text{col}}$ (pp) on swap-invariant + POPE + MME; **Pareto dominance** over *tuned* global-2×/CAST/DMAS/PAI/TLVS-style in the (recovery, collateral) plane; provenance effect C2−C3 with C3−C1 and C0m controls; localization curve ($K$ vs recovery vs random, split-clean); gate AUROC on the swapped split; eager-attention latency/memory overhead.

This resolves the earlier S1↔F2 contradiction: S1 is stated as recovery of the **C2** gap, and F2 (role falsified) is defined on the **C2−C3** contrast — the ceiling and the mechanism test now reference the same condition.

---

## 6. Ablations (each isolates one component)

1. **Gate-variable dissociation** (C1/C2/C3/C0m + PS) — role vs boundary vs sink vs recency. Expect C2−C3>0, C3−C1≈0, C0m≈C1; else Fallback A.
2. **Localization vs global** (split-clean top-$K$ vs all-heads global-2×) — expect targeted ≥ global recovery at strictly lower collateral, beats random.
3. **Head-set-size sweep** ($K\in\{3,10,30,50,100\}$) — recovery knee at $K\le50$ (guards Dual-Pathway redundancy `2605.13156`).
4. **Per-instance gate** (gated vs always-on-targeted vs always-on-global; + TLVS-style per-token strength as a comparator) — gated preserves swap-invariant/POPE/MME at matched recovery.
5. **Steering rule/target** — per-instance $\hat p(x)$ vs population-mean clamp vs fixed-2× vs additive; **independent vs joint/iterative solve**; report downstream divergence from true C2.
6. **Intervention timing** (re-look-only vs all-steps) — re-look-only suffices at lower collateral.
7. **Gate feature set** (internal activations vs surface features vs **answer-margin-only**) — internal features add AUROC over the margin-only baseline.

---

## 7. Differentiation (stress-test Q1)

**Method-property columns** (evaluation/design contributions listed separately below). Cells are phrases, not ticks.

| Work | training-free | provenance-conditioned (user-vs-self) | per-instance **abstain** gate | re-examination illusion |
|---|---|---|---|---|
| VRRL `2607.02490` | no (RL) | no | no (uniform) | symptom only, via RL |
| Phenomenon `2605.15864` | 2× PoC | no | no (all heads, always-on) | diagnoses it; global fix |
| CAST `2605.04641` | yes | no (caption-salience) | no (always-on) | no |
| DMAS `2602.21704` | yes | no (input-conditional **vector choice**, no abstain) | no | no |
| VisFlow `2506.12609` | yes | no (token-**type**: system/visual/text) | no | no |
| TLVS `2606.07647` | calibration-trained | no (per-**token** strength) | no (no abstain) | no |
| PAI `2407.21771` | yes | no (global image-attn amplification) | no | no |
| Gaze Heads `2606.14703` | yes | no (no provenance dim) | no | no |
| **Ours** | **yes** | **yes (self-vs-user provenance of the re-look trigger)** | **yes (abstain)** | **yes** |

*Evaluation/design contributions (separate from method properties):* the matched-content **causal dissociation** of provenance vs position vs boundary (C2/C3/C0m/PS), and the **collateral-on-swap-invariant** joint metric. The single load-bearing novel method atom is **provenance-conditioned, per-instance-abstaining restoration of visual-routing heads at re-look steps** — defended above against each neighbor individually.

---

## 8. Failure modes, fallbacks, limitations (stress-test Q3–Q9)

- **Dominant confounds & Q3:** the gap may be **recency** (C4 re-inserts $I_b$ recently; IKOD/`2508.17807`) or **turn-boundary/sink** (user turn = boundary + sink tokens). Controls: C2 (user, far-back, no re-insert), C0m (markers, empty $U$), PS (position sweep). These *are* the **Week-1 GO/NO-GO** and the C2−C3/C3−C1/C0m tests.
- **Interaction non-identifiability (conceded):** provenance×boundary is not point-identified (no natural user-far-back-$\beta$0 cell). We report a descriptive decomposition under a stated caveat and pre-commit: if C0m or C3 explains the C2−C3 gap → **Fallback A**.
- **Fallback A (pre-committed):** role falsified but a training-free trigger works → reframe to the winning variable. C3/C0m surge → **turn-boundary/sink-reset** fix (inject a boundary marker at re-look, gated). PS drives it → **image-recency** fix (re-prefill/re-position visual KV at re-look). Both remain training-free, targeted, gated, still beat naive global-2× on collateral.
- **Fallback B (minimum guaranteed contribution, pre-committed):** if no clean gate variable or H2/H3/F1 fail — the paper is *the first head-level, matched-content causal dissociation of what does and does-not drive the re-examination surge* (provenance vs position vs boundary vs content), strictly deeper than the phenomenon paper's layer-level analysis, plus the strongest training-free targeted fix Pareto'd vs global-2×/CAST/DMAS. **We commit to this single minimum contribution clearing ICLR's bar** on the grounds that no prior work performs the matched-content head-level dissociation (`2605.15864` is layer-level; `2606.14703` has no provenance dimension; CAST/DMAS/VisFlow do no dissociation). This is honest and independent of whether the role hypothesis or the fix survives.
- **Q7 — infinite compute:** an *absolute* accuracy fix on a latent capability the model fails to deploy, not a compute trade-off; the gap is not sampling luck.
- **Limitations:** VisualSwap is 800 math-centric pairs → we pre-register **power/MDE** and an N-augmentation trigger (§ plan) before claiming "no degradation"; headline effects are largest on >48GB Thinking models → we must reproduce a $\ge$15pp C2 gap at 8B (30B-A3B FP8 is a stretch); provenance heads may be model-specific (per-model localization; cross-model transfer is a stretch); CAST/DMAS have no public code → faithful reimplementation with **matched-budget strength tuning** (§ plan) and POPE home-turf sanity before cross-eval; the population-mean-clamp / independent-solve ceiling may cap recovery (measured in §6.5).

---

## 9. Reproducibility

Frozen open models (Qwen3-VL-8B-Thinking/Instruct, Qwen2.5-VL-7B-Instruct, +1 open reasoning-7B; LLaVA-1.5-7B & InternVL2.5-8B controls), public datasets (VisualSwap secured on HF), eager-attention hooks, and a supervised probe. Every number traces to `results/`. Per run: git commit, seed, config YAML, GPU model (4×L40-48GB), library versions, model revision hash (integrity rule 4). Steering is deterministic given captured attention and a sampling seed.
