# Phase 4 Failure Diagnosis (PPO Underperformance)

This diagnosis is read-only on completed artifacts. No training config changes were made.

Supporting outputs generated from completed runs:
- `results/phase4_failure_diagnosis/ppo_full_dcr_curves.png`
- `results/phase4_failure_diagnosis/failure_diagnosis_stats.json`

## 1) Convergence Check (ppo_full seeds, 5M-step curves)

Method:
- Used `eval_dcr_mean` vs `step` from each `results/phase3_full/curves/ppo_full/ppo_full_seed_<seed>_curve.csv`.
- Fit a linear slope on the last 20% of eval points.
- Reported slope in DCR points per 1M steps.

Results:
- Seed 11:
  - Final eval DCR: `74.73`
  - Last-20% slope: `+14.09 DCR / 1M steps`
  - Last-20% endpoint delta: `-2.67`
- Seed 22:
  - Final eval DCR: `96.67`
  - Last-20% slope: `+13.27 DCR / 1M steps`
  - Last-20% endpoint delta: `+25.38`
- Seed 33:
  - Final eval DCR: `47.49`
  - Last-20% slope: `+15.95 DCR / 1M steps`
  - Last-20% endpoint delta: `-14.04`

Interpretation:
- None of the three seeds shows a clean flat plateau in the final 20%.
- All seeds have positive fitted tail slopes, but two seeds have volatile end-segment drops.
- This pattern indicates **high variance / unstable convergence**; 5M steps did not produce uniformly stable asymptotes across seeds.

## 2) The ~97% Seed (Reachability vs Reliability)

Best seed identification:
- Best `ppo_full` seed is **seed 22** (training final eval DCR `96.67`; Tier-1 mean DCR `94.90`).

Per-seed Tier-1 comparison (`ppo_full_seed{11,22,33}_T1_episodes.csv`):
- Seed 11:
  - DCR mean: `78.26`
  - TCU mean: `1223.64`
  - CER mean: `12.4567`
- Seed 22 (best):
  - DCR mean: `94.90`
  - TCU mean: `13765.80`
  - CER mean: `0.00737`
- Seed 33:
  - DCR mean: `39.93`
  - TCU mean: `47026.31`
  - CER mean: `0.00234`

What the good seed did differently:
- It reached high compliance, but **not** by chemical efficiency.
- Seed 22 achieves high DCR with much higher chemical usage than seed 11.
- Across the three seeds, performance appears reachable but **unreliable and policy-style dependent** (seed 11 conservative vs seed 22 aggressive vs seed 33 pathological over-dosing).

Conclusion on reachability:
- High performance is reachable (seed 22), but current setup yields low reliability and large seed-to-seed policy divergence.

## 3) Chemical Waste Mechanism (Best PPO vs PID behavior)

Question tested:
- Is PPO dosing unnecessarily while already in-band?

Method:
- Action-audit rollouts in the same simulator setting (Tier-1 style), comparing:
  - best PPO full seed (seed 22), and
  - deadband PID baseline.
- Measured fraction of in-band timesteps with non-null action.

Results (120 episodes audit):
- Best PPO seed (22):
  - In-band steps: `54,479`
  - In-band dose steps: `49,619`
  - In-band dosing fraction: **`0.9108`** (91.08%)
- PID:
  - In-band steps: `56,470`
  - In-band dose steps: `219`
  - In-band dosing fraction: **`0.00388`** (0.388%)

Action distribution signal:
- PPO seed 22 uses non-null actions heavily across both acid and alkaline bins.
- PID is overwhelmingly action 0 (`~97.66%` null overall in this audit), with sparse corrective dosing.

Interpretation:
- PPO frequently doses even when compliant, strongly consistent with observed high TCU and poor CER.
- This is the central behavioral failure mode behind “DCR not terrible but chemistry waste extreme.”

## 4) Entropy / Stability

Requested source:
- “From training curves, report final entropy per seed.”

Constraint:
- Current saved PPO curve schema (`step`, `rollout_reward_mean`, `rollout_reward_std`, `eval_dcr_mean`) does **not** include entropy.
- Therefore final entropy cannot be read directly from stored training curves.

Post-hoc proxy (final models, reset-state entropy estimate):
- Seed 11: mean entropy `0.271` (sd `0.360`)
- Seed 22: mean entropy `1.575` (sd `0.558`)
- Seed 33: mean entropy `0.635` (sd `0.451`)

Interpretation:
- Best seed 22 remains relatively stochastic versus other seeds, suggesting the “high DCR” solution is not a sharp deterministic settled policy.
- Combined with seed variance, this supports a stability/reliability concern rather than a single clean converged attractor.

## Ranked, Principled Reasons PPO Underperformed

1. **In-band null-action failure (primary)**
   - Evidence: PPO doses on ~91% of compliant timesteps vs PID ~0.39%.
   - Consequence: very high TCU and low CER, even when DCR can be high.

2. **Seed-dependent policy mode collapse/divergence**
   - Evidence: `ppo_full` seed outcomes span ~40% to ~95% Tier-1 DCR with huge TCU spread.
   - Consequence: good policy is reachable but not reliably found.

3. **Late-training instability / non-plateau at 5M**
   - Evidence: tail slopes remain positive and endpoint behavior is volatile in 2/3 seeds.
   - Consequence: run horizon did not guarantee stable convergence for all seeds.

4. **Objective trade-off imbalance in practice (compliance vs chemistry)**
   - Evidence: best DCR seed still has very high TCU and extremely poor CER compared to PID/rule/LUT.
   - Consequence: learned policy exploits compliance without discovering economical null-heavy control.

5. **Observation noise + stochastic policy combination may sustain unnecessary dithering**
   - Evidence: post-hoc entropy for best seed remains relatively high.
   - Consequence: persistent random actuation can maintain over-dosing once near-band.

## Scope / Non-actions

- No training config was altered.
- No additional retraining was performed.
- This document is diagnosis-only to guide next-step hypothesis testing.
