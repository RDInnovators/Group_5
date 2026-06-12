# Phase 3 Critic Learnability Probes (A/B/C)

Goal for this step: raise critic learnability (explained variance), not optimize DCR.

All probes were 100k-step diagnostics on seed 11, GPU, one change axis at a time, with advantage normalization kept on.

Primary artifact:

- `results/phase3a_singlecheck/phase3_critic_fix_stats.json`

Probe curves:

- `results/phase3a_singlecheck/ppo_seed11_probeA_rewardscale_100k_curve.csv`
- `results/phase3a_singlecheck/ppo_seed11_probeB_gamma095_100k_curve.csv`
- `results/phase3a_singlecheck/ppo_seed11_probeC_bigcritic_vf1_100k_curve.csv`

---

## Probe A — reward/return scaling

Change:

- Scale training reward by `0.1` (divide by 10)
- Keep gamma `0.99`, default critic architecture, value-loss coef `0.5`

Results:

- Explained variance (first / mean / last): **0.0068 / 0.0440 / 0.2624**
- Value-target std (mean): **3.15** (down from prior ~28.6 scale)
- DCR (first -> last): **40.83% -> 17.92%** (decline)

Interpretation:

- This is the only probe that moved EV materially positive and trending up by run end.
- Still below target EV > 0.3 on mean, but much better than baseline near-zero behavior.

## Probe B — shorter effective horizon

Change:

- From Probe A base, set gamma `0.95`

Results:

- Explained variance (first / mean / last): **0.0113 / 0.00043 / -0.0093**
- Value-target std (mean): **1.77**
- DCR (first -> last): **41.23% -> 25.53%** (decline)

Interpretation:

- Lower gamma did not improve critic fit; EV remained near zero/negative overall.

## Probe C — value-loss capacity

Change:

- From Probe B base, increase value-loss coefficient (`vf_coef=1.0`) and use larger separate critic network

Results:

- Explained variance (first / mean / last): **-0.2015 / -0.2162 / -0.2048**
- Value-target std (mean): **1161.23** (exploded)
- DCR (first -> last): **25.98% -> 20.58%** (decline)

Interpretation:

- This setting is unstable; critic training diverged badly and EV became strongly negative.

---

## Summary and recommendation

Best critic-learnability result among tested probes:

- **Probe A (reward scaling 0.1, gamma 0.99, vf_coef 0.5, default critic)**  
  - Highest EV mean (**0.0440**) and positive EV trend to **0.262** by end.

What this means:

- None of A/B/C achieved the target “clearly >0” regime (e.g., EV > 0.3 mean), so critic is improved but not fully learnable yet.
- Probe A is the only direction that helps; B and especially C hurt critic learnability.

Recommended configuration to keep for next iteration (pending your review):

- reward scaling `0.1`
- gamma `0.99`
- value-loss coefficient `0.5`
- default critic architecture
- advantage normalization `on`

No full multi-seed run was launched.
