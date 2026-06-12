# Phase 3a Interim Health Check (PPO Seed 11)

This is an early, seed-level health check requested while the full Phase 3a run continues in the background. No training was interrupted.

## Data source used

- Curve file: `results/phase3a/curves/ppo_seed_11_curve.csv`
- Derived stats: `results/phase3a/curves/ppo_seed_11_interim_stats.json`
- Interim plot: `results/phase3a/curves/ppo_seed_11_interim_plot.png`

## 1) Seed-11 training curve summary

- Total logged PPO updates: **2442** rows.
- DCR evaluation points (every ~100k steps): **50**.
- First eval DCR (at step 100,352): **20.89%**.
- Last eval DCR (at step 5,000,000): **1.62%**.
- Final-window DCR mean:
  - Last 5 evals: **1.67%**
  - Last 10 evals: **1.63%**
- Rollout reward mean:
  - First 200 updates: **0.00158**
  - Last 200 updates: **0.00331**

## 2) Is PPO learning on seed 11?

Plain reading of the curve: **PPO is not learning a useful control policy on this seed**.

- DCR is not rising and stabilizing; it declines from an early ~20% level to ~1.6% and stays there.
- The final DCR window is very low and flat (~1.6%), which indicates collapse/degenerate behavior rather than improvement.
- Reward mean shows only a very small change and does not correspond to improved compliance behavior.

**Final-window mean DCR reached in training (seed 11): 1.63% (last 10 eval points).**

## 3) Competitiveness vs baselines

Using prior Phase 2 baseline references:

- Deadband PID: ~97% DCR
- LUT: ~90% DCR
- PPO seed 11 interim: **~1.63% DCR**

Interim verdict: **far below competitive range** (not close to LUT or PID).

## Note

This is an interim single-seed diagnostic only. The full Phase 3a run is still continuing for remaining seeds as requested.
