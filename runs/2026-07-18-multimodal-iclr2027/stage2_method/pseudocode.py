"""
Pseudocode (NOT runnable) for the role-gated, training-free visual re-engagement fix.
Frozen VLM M. Eager attention (per-head capture). No weight updates, no RL.
Grounded in method.md; implemented in Stage 3.
"""

# ----------------------------------------------------------------------------
# PART A — Causal dissociation: what gates the re-examination surge?
# ----------------------------------------------------------------------------
def matched_content_dissociation(M, visualswap_sensitive):
    """Boundary-MATCHED dissociation. BYTE-IDENTICAL U; image CONTENT fixed (re-encoded I_b);
    total measurement-query length + image->query distance ALIGNED across C1/C2/C3/C0m so the
    position basis is not silently varied by user-wrapper tokens. Primary role test = C2 vs C3
    (matched boundary). C0m = marker/sink control. PS = recency curve. C4 = recency-inclusive oracle."""
    U = "let me look at the image again"          # fixed across all conditions
    records = []
    for (I_b, Q, R_a, y) in visualswap_sensitive:
        s1  = run_capture(M, image=I_b, ctx=[Q, R_a], inject=U, role="assistant",
                          image_pos="original_farback", boundary=False)     # C1 illusion baseline
        s2  = run_capture(M, image=I_b, ctx=[Q, R_a], inject=U, role="user",
                          image_pos="original_farback", boundary=True, pad_align=True)  # C2 user, far-back
        s3  = run_capture(M, image=I_b, ctx=[Q, R_a], inject=U, role="assistant",
                          image_pos="original_farback", boundary=True, pad_align=True)  # C3 assistant boundary
        s0m = run_capture(M, image=I_b, ctx=[Q, R_a], inject="", role="user",           # C0m markers, EMPTY U
                          image_pos="original_farback", boundary=True, pad_align=True)   #    -> sink-token control
        s4  = run_capture(M, image=I_b, ctx=[Q, R_a], inject=U, role="user",
                          image_pos="recent_reinserted", boundary=True)     # C4 oracle (recency-inclusive)
        ps  = [run_capture(M, image=I_b, ctx=[Q, R_a], inject=U, role="assistant",
                           image_pos=p, boundary=False) for p in POSITION_GRID]  # PS recency sweep
        records.append(dict(c1=s1, c2=s2, c3=s3, c0m=s0m, c4=s4, ps=ps, label=y))

    # DESCRIPTIVE decomposition under stated no-interaction caveat (interaction NOT point-identified).
    delta = per_head_contrast(records, "c2", "c3")            # provenance-differential (matched boundary)
    prov_ci = instance_bootstrap_ci(records, effect="c2_minus_c3")   # primary role test
    bnd_ci  = instance_bootstrap_ci(records, effect="c3_minus_c1")   # boundary control (want ~0)
    sink_ci = instance_bootstrap_ci(records, effect="c0m_minus_c1")  # marker/sink control (want ~0)
    delta = fdr_correct(delta, alpha=0.05)                    # across ~1150 heads (BH)
    return delta, dict(prov=prov_ci, boundary=bnd_ci, sink=sink_ci, records=records)  # H1 lives here

# ----------------------------------------------------------------------------
# PART A (cont.) — localize role-conditioned visual-routing heads
# ----------------------------------------------------------------------------
def localize_heads(M, delta, records, K_grid=(3, 10, 30, 50, 100)):
    """SPLIT-CLEAN (de-circularized): select on localization split, score on disjoint eval split."""
    loc_split, eval_split = nested_cv_split(records)  # disjoint instance folds, pre-registered
    role_rank = argsort_desc(abs(delta))              # provenance-differential ranking (from C2-C3, FDR)
    def causal_effect(H, split):
        return recovery_when_only_steering(M, split, heads=H)
    curves_loc = {K: causal_effect(topK(role_rank, K), loc_split) for K in K_grid}
    H_K = select_knee(curves_loc)                     # knee chosen on LOCALIZATION split
    # report recovery + random control on the DISJOINT eval split:
    rec_eval  = causal_effect(H_K, eval_split)
    rand_eval = [causal_effect(random_heads(len(H_K)), eval_split) for _ in range(20)]
    stability = headset_stability_across_folds(role_rank, records)   # report fold agreement
    # per-INSTANCE engaged target predictor (not a population-mean clamp):
    p_hat = fit_target_predictor(M, loc_split, heads=H_K, condition="c2")  # x -> per-head C2 mass
    return H_K, p_hat, dict(rec_eval=rec_eval, rand_eval=rand_eval, stability=stability)  # H2 lives here

# ----------------------------------------------------------------------------
# PART B — the steering operator (training-free, at re-look steps only)
# ----------------------------------------------------------------------------
def rescale_image_attention(pre_softmax_logits, image_idx, target_mass):
    """Additive per-head shift on image-key logits so post-softmax image mass == target.
    Uniform over image keys, zero elsewhere => within-image and within-text relative
    patterns preserved; only image-vs-text balance moves. c solved in closed form / bisection."""
    def image_mass(c):
        z = pre_softmax_logits.clone(); z[image_idx] += c
        return softmax(z)[image_idx].sum()
    c = solve_for_shift(image_mass, target=target_mass)        # 1-D monotone in c
    z = pre_softmax_logits.clone(); z[image_idx] += c
    return softmax(z)

def steered_decode(M, x, H_K, p_hat, gate, tau, mode="independent"):
    """Attention hook fires only at re-look-span steps AND only if the gate fires.
    Target is PER-INSTANCE p_hat(x) (not a population-mean clamp). mode='joint' iteratively
    re-solves so composed downstream masses approach the C2 joint state (ablation §6.5)."""
    fired = gate(x) > tau                                       # per-instance abstain decision
    targets = p_hat(x)                                          # per-head, per-instance target mass
    def hook(layer, head, step, logits, image_idx):
        if fired and is_relook_step(step) and (layer, head) in H_K:
            return rescale_image_attention(logits, image_idx, targets[(layer, head)])
        return logits                                          # else: vanilla
    out = M.generate(x, attention_logit_hook=hook, joint_resolve=(mode == "joint"))  # frozen weights
    return out

# ----------------------------------------------------------------------------
# PART B (cont.) — the tiny per-instance ABSTAIN gate g(x) (supervised, NO RL)
# ----------------------------------------------------------------------------
def train_gate(M, H_K, p_hat, source_heldout, swapped_heldout, ood_set):
    """Label on UNSWAPPED source split = re-look-BENEFIT PROXY (no I_b there). Deployment-AUROC
    is reported on a held-out SWAPPED VisualSwap split + a non-math OOD set (anti-shortcut)."""
    X, Y = [], []
    for x in source_heldout:                        # proxy labels (no swap available here)
        phi = cached_features(M, x, H_K, p_hat)     # pre-trigger visual-mass vs target + answer-logit margin
        y_self   = correct(decode_noop(M, x))
        y_relook = correct(user_turn_relook(M, x))  # user-role re-look benefit (proxy, no I_b)
        X.append(phi); Y.append(int((y_relook == 1) and (y_self == 0)))
    g = fit_logistic_or_mlp(X, Y)                   # tiny probe on frozen features
    tau = calibrate_threshold(g, X, target_intervention_budget=0.3)
    # H4/S5 evaluation on the ACTUAL deployment distribution + OOD, vs a margin-only baseline gate:
    auroc_swapped = auroc_with_ci(g, swapped_heldout)   # true deployment distribution
    auroc_ood     = auroc_with_ci(g, ood_set)           # non-math anti-shortcut
    auroc_margin  = auroc_with_ci(answer_margin_gate, swapped_heldout)  # baseline to beat
    return g, tau, dict(swapped=auroc_swapped, ood=auroc_ood, margin_baseline=auroc_margin)

# ----------------------------------------------------------------------------
# Orchestration (maps to compute phases P1..P6 in experiment_plan.yaml)
# ----------------------------------------------------------------------------
def run_study(M, data):
    delta, ci = matched_content_dissociation(M, data.swap_sensitive)                     # P1: WEEK-1 GATE
    # GO iff self-vs-C2 gap>=15pp AND (C2-C3 excludes 0) AND (C3-C1 small) AND (C0m does not reproduce surge):
    assert_go_no_go(gap_self_vs_C2(M, data) >= 15,
                    ci["prov"].excludes_zero(), ci["boundary"].is_small(), ci["sink"].is_small())  # else Fallback A/B
    report_mde_and_f3_collateral(M, data)                                                # W1: MDE + global-2x collateral (F3)
    H_K, p_hat, loc = localize_heads(M, delta, data.records)                             # P2 (split-clean)
    g, tau, auroc = train_gate(M, H_K, p_hat, data.source_heldout, data.swapped_heldout, data.ood_set)  # P3/P4
    tuned = tune_each_baseline_on_dev(M, data.dev)   # P5: matched-budget strength tuning (fair Pareto)
    results = evaluate_grid(                                                             # P5/P6
        M, data,
        methods=["self", "C2_ceiling", "C4_oracle", "global_2x*", "CAST*", "DMAS*",
                 "PAI*", "TLVS_pertoken*", "VCD", "targeted_alwayson", "targeted_gated(ours)"],  # * = tuned
        # primary Rec scored vs C2 ceiling; headline = absolute accuracies with paired bootstrap CIs
        metrics=["acc_self", "acc_C2", "acc_C4", "acc_method", "Rec_vs_C2",
                 "collateral_swap_invariant", "POPE_sub", "MME_sub"],
        seeds=[0, 1, 2], ci="joint_seed_instance_bootstrap", tuned=tuned)
    return results   # every number traces to results/ (integrity rule 1)
