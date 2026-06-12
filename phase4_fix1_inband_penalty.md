# Phase 4 Fix #1: In-Band Dosing Penalty

Goal of this fix:
- Apply one principled reward change only: **when pH is already compliant, non-null dosing should be penalized**.
- This encodes controller logic correctness ("if compliant, stop dosing"), independent of whether PPO wins.
- No other reward terms were changed.

## What was changed

In `water_methodology_impl.py`:
- Added new reward term weight:
  - `W_INBAND_DOSE = -1.0`
- Added term in `WastewaterMDP.step()`:
  - `Rinband = 1.0 if in_comp and action != 0 else 0.0`
  - Reward now includes `+ W_INBAND_DOSE * Rinband`.

Unchanged (locked):
- `W_COMP, W_DEV, W_DOSE, W_OVER, W_ESC = 3.0, -1.0, -0.3, -0.5, -0.1`
- Reward scale, gamma, vf_coef, critic, adv-norm, all training hyperparameters.

## Pre-retrain sanity checks (cheap)

### A) In-band immediate reward ordering (pH=7.5)

Observed one-step rewards by action:
- `a0(null)=3.0`
- best non-null was `a5=2.8` (others lower)

Result:
- **Null is now strictly highest-reward in-band**.

### B) Episodic reward ranking: efficient vs wasteful policy

60-episode cheap comparison:
- Deadband PID mean total reward: `1651.76 ± 94.68`
- Always-max-dose policy mean total reward: `840.21 ± 59.22`

Result:
- **Deadband PID clearly outranks always-max-dose** under reward objective.
- This confirms in-band waste is now penalized meaningfully.

## Retrain + evaluation plan executed

No further reward modifications were made.

To avoid prior long-lived-process issues, retraining is isolated per seed:
- Added `train_one_job_fix1.py` (single-seed PPO-full, locked config, live curve writing).
- Added `phase4_fix1_run.py`:
  - sequential seeds `11,22,33`
  - full budget `5,000,000` steps each
  - Tier-1 evaluation (500 episodes/seed)
  - metrics written to `results/phase4_fix1/phase4_fix1_eval_summary.json`.

Detached launch:
- Running via `nohup` to survive disconnect.
- Log: `results/phase4_fix1_run.log`.

## Current run status (at write time)

Observed running processes:
- `phase4_fix1_run.py` orchestrator active.
- `train_one_job_fix1.py --seed 11` active.

This file is intentionally interim until retrain+eval finish.

## What will be reported after completion (no extra reward changes)

Per-seed and mean±SD (requested):
- DCR, TCU, CER, in-band dosing fraction for retrained PPO-full (Tier-1, 500 eps).
- Comparison to PID reference.

No additional reward edits will be made before review.
