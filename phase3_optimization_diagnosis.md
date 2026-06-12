# Phase 3 Optimization-Dynamics Diagnosis (No Full Retraining)

This report targets optimization path issues (critic/advantages), with no full run launched.

Primary artifact:

- `results/phase3a_singlecheck/phase3_optimization_diagnosis_stats.json`
- Curve with diagnostics: `results/phase3a_singlecheck/ppo_seed11_optdiag_100k_curve.csv`

## 1) Value function fit (leading hypothesis)

Measured in a 100k-step PPO diagnostic run (seed 11, same environment/reward, GPU):

- Value loss (`vf_loss_mean`) starts high and decreases:
  - first: **1093.84**
  - last: **247.28**
  - min observed: **183.82**
  - max observed: **1093.84**
- Explained variance of critic predictions on value targets is effectively zero:
  - first: **0.00068**
  - last: **-0.00187**
  - mean: **0.00133**

Interpretation:

- The critic loss decreases numerically, but explained variance remains ~0/negative.
- That means the value function is still not explaining return variance (critic is effectively uninformative), so advantage estimates are low-quality/noisy for policy improvement.

## 2) Return/advantage scale

From the same diagnostic run:

- Episode return variability (within rollout episodes):
  - mean std across rollouts: **277.22**
- Value-target variability (`ret`):
  - mean std across rollouts: **28.57**
- Raw advantage variability:
  - mean std across rollouts: **28.55**
- Advantage normalization status:
  - **Enabled** in current PPO path (`adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)`)
  - in probe: raw advantage std ~28.55, used advantage std ~1.0 (as expected)

Interpretation:

- Returns and targets are high-variance.
- Advantages are already normalized before policy update, so “missing advantage normalization” is **not** the issue.

## 3) Reward magnitude interaction with long horizon

Observed interaction signals:

- Per-step rewards are O(1), but long horizon + mixed starts produce wide episodic return spread (std ~277).
- Critic target spread is substantial (target std ~28.6), while explained variance stays ~0.
- This indicates critic fit is struggling relative to target variability, despite normalization of policy advantages.

Interpretation:

- The issue is consistent with critic learning quality / target-noise structure, not reward sign bug.
- Large-horizon/high-variance returns appear to be difficult for current critic setup to model accurately.

## 4) Quick isolation test (~100k) with advantage normalization ON

Requested directional probe run:

- Run: 100k PPO, seed 11, advantage normalization explicitly ON.
- DCR trend:
  - first eval DCR: **42.27%**
  - last eval DCR: **19.51%**
  - last-3 mean DCR: **20.23%**
- Entropy trend:
  - first eval entropy: **2.3753**
  - last eval entropy: **2.3709**
  - remains high (near-uniform over 11 actions).

Directional result:

- Even with advantage normalization enabled, DCR still moves in the wrong direction (declines).
- Entropy does not collapse; policy remains highly stochastic.

## Plain verdict

- The optimization diagnosis supports the critic-quality hypothesis:
  - value explained variance is near zero/negative throughout, so the critic is not learning a useful signal.
- Missing advantage normalization is ruled out (it is already enabled).
- Failure mode is likely in critic target fitting / variance structure and downstream advantage quality, not reward correctness or entropy collapse.
