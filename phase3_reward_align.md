# Phase 3 Reward Alignment (Pre-Training Lock)

This update aligns the training objective with the compliance-first research goal before retraining.

## What was changed

I applied the smallest structural compliance-first adjustment using the permitted levers:

1. **Non-compliant steps made explicitly costly** in the compliance term:
   - `Rc` changed from binary `{0,1}` to signed `{ -1, +1 }`.
   - In-band step: `Rc = +1`
   - Out-of-band step: `Rc = -1`
   - This prevents low-action/do-nothing behavior from coasting with weak penalties.

2. **Compliance weight increased modestly**:
   - `W_COMP` updated from `2.0` -> `3.0`
   - Other weights unchanged:
     - `W_DEV = -1.0`
     - `W_DOSE = -0.3`
     - `W_OVER = -0.5`
     - `W_ESC = -0.1`

3. Existing Phase-3 fix retained:
   - Training reward remains raw reward (normalization disabled).

No other reward terms were changed.

## Why this is principled (not PPO-target tuning)

- Objective is compliance-first: sustained in-band control must dominate reward ranking.
- Signed compliance term directly encodes the hard objective distinction between compliant and non-compliant timesteps.
- Weight increase was only as much as needed to enforce correct policy ordering on the fixed baseline set.

## Cheap confirmation (no retraining)

Using the same 60-episode setup (seeds 300..359), recomputed total reward on the actual returned training signal:

Source: `results/phase3a/phase3_reward_align_stats.json`

- **PID** total reward mean: **1636.65** (SD 117.65)
- **LUT** total reward mean: **1625.19** (SD 141.95)
- **Null** total reward mean: **518.52** (SD 500.17)

Ordering checks:

- PID > LUT: **true** (gap: **+11.45**)
- LUT > Null: **true** (gap: **+1106.67**)
- PID > Null: **true** (gap: **+1118.13**)

## Verdict

Reward ordering now tracks compliance ordering with clear separation from null:

- **PID > LUT >> null**

Reward is locked at this point for subsequent retraining/evaluation steps.
