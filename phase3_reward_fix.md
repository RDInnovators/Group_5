# Phase 3 Reward Fix (Pre-Retrain)

## Change made

I disabled running reward normalization in `WastewaterMDP.step()` so PPO now trains on the **raw reward**.

- File changed: `water_methodology_impl.py`
- Behavior before: returned reward was `rn = reward_norm.norm(r)` (running z-score)
- Behavior now: returned reward is `train_r = r` (raw reward), with terminal penalty unchanged

### Retained-but-disabled note

The normalization code is preserved in-place and labeled:

- `DISABLED (Phase 3 reward-fix): running reward normalization layer`

Reason documented in code:

- Running normalization inverted policy ordering on the optimized signal (null > PID), causing objective mismatch and collapse.

## Reward terms/weights changed?

No.  
Per instruction, reward terms and weights are unchanged (`W_COMP`, `W_DEV`, `W_DOSE`, `W_OVER`, `W_ESC` unchanged). Only the post-hoc normalization layer on the returned training signal was removed.

---

## Cheap confirmation on the **actual training signal** (now raw)

I re-ran total episode reward comparisons using the environment return (`env.step(...)->r`) after the fix, over 60 episodes (seeds 300..359):

Source: `results/phase3a/phase3_reward_fix_stats.json`

- Deadband PID total reward (mean ± SD): **1189.51 ± 113.92**
- LUT total reward (mean ± SD): **1246.13 ± 142.15**
- Null total reward (mean ± SD): **1094.52 ± 254.86**

Ordering checks:

- PID > null: **true**
- LUT > null: **true**
- PID >= LUT: **false** (LUT > PID on this reward aggregate)

## Interpretation

- The key failure mode is fixed: higher-compliance controllers (PID/LUT) now both outscore null on the training objective signal.
- The exact PID vs LUT ordering is **LUT > PID** in this test window; so the observed ordering is:
  - **LUT > PID > null**
- This still satisfies the critical requirement that null is no longer preferred by the optimized objective.

## Status

No retraining was run in this step. This is a pre-retrain fix + objective-order sanity check only.
