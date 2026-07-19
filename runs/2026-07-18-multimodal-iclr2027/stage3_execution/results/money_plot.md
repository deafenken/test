# The money plot — provenance vs magnitude (generation-based accuracy, real)

**Qwen3-VL-8B-Thinking, VisualSwap-MathVerse letter subset, N=85.** Generation-based accuracy (generate the answer,
extract the option letter, exact-match vs A_b). global-2× and mag-match apply attention amplification DURING generation.
Code: `../code/scripts/vsbench_eval.py`.

| method | accuracy | 95% CI | recovery vs self |
|---|---|---|---|
| base (sees I_b fresh; ceiling) | 0.600 | [0.494, 0.706] | — |
| **self (illusion floor)** | **0.012** | [0.000, 0.035] | — |
| user (multi-turn ceiling) | 0.541 | [0.435, 0.647] | +0.529 |
| **prov_fix — provenance injection (OURS)** | **0.541** | [0.435, 0.647] | **+0.529** |
| global-2× (SOTA attention amplification) | 0.035 | [0.000, 0.082] | +0.024 |
| mag-match (our amp, magnitude-matched) | 0.059 | [0.012, 0.118] | +0.047 |

## Findings
1. **Provenance is the lever, magnitude is not (decisive).** Provenance re-framing (prov_fix) recovers **+52.9 pp**, and
   **exactly matches the manual user-turn ceiling (both 0.541)** — an automatic, training-free method reaching the oracle.
   Attention amplification — global-2× (+2.4 pp) and the magnitude-matched control (+4.7 pp) — barely moves accuracy.
   A **>10× gap**. This is the paper's core mechanistic result.
2. **Our automatic provenance-injection == the user-turn oracle.** No human second turn needed; auto-reframing the
   model's own re-look as a user turn achieves the same recovery.

## HONEST caveat (do not overclaim a leaderboard beat yet)
- Our global-2× recovers only **+2.4 pp here, NOT the paper's +18.2 pp**. Reason: our short forced-letter accuracy sits
  at a much harsher operating point (self ≈ 1.2% vs their probe ≈ 29.5%; base 0.60 vs their 0.83). At self≈1% there is
  almost no room for amplification to help, whereas provenance re-framing restarts the turn.
- Therefore the **mechanism claim (provenance ≫ magnitude) is strongly supported**, but a **direct leaderboard claim
  ("beats their 54.8") is NOT yet valid** — that needs their exact protocol (full R_b generation + LLM-judge accuracy
  on the 3 scaffolds), which will place all methods at the same operating point as their Tables 4/9.
- The constrained self-scaffold (forced letter mid-continuation) may also inflate the illusion gap; the protocol-matched
  run will settle both the gap size and global-2×'s true recovery in our hands.

## Next
- Protocol-matched reproduction (full R_b + judged accuracy) to reproduce their base/probe/global-2× numbers exactly,
  then place prov_fix / PACD / mag-match on the same axis (the true money plot).
