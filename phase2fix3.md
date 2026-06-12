# Phase 2 Fix 3 — TCU Diagnostic (Pre-Phase 3)

## Objective

Diagnose whether high PID `TCU` is a real controller behavior or a metric/accounting artifact.

No controller tuning or retraining was performed.

## 1) Action usage diagnostics (few-episode sample)

Executed over 12 episodes per controller (shared simulator and start-state distribution).

### Non-null action fraction

- Rule-based: **0.14965277777777777**
- PID: **0.4029513888888889**
- LUT: **0.09479166666666666**
- Null: **0.0**

PID takes non-null actions ~40.3% of timesteps, versus ~15.0% (rule-based) and ~9.5% (LUT).

### Action magnitude distributions (fraction by action index)

#### Rule-based
- `0`: 0.8503
- `3`: 0.0141
- `8`: 0.1356

#### PID
- `0`: 0.5970
- `1`: 0.0938
- `2`: 0.0188
- `3`: 0.0075
- `4`: 0.0054
- `5`: 0.0139
- `6`: 0.1396
- `7`: 0.0313
- `8`: 0.0175
- `9`: 0.0031
- `10`: 0.0722

#### LUT
- `0`: 0.9052
- `3`: 0.0005
- `5`: 0.0023
- `8`: 0.0833
- `10`: 0.0087

#### Null
- `0`: 1.0

Interpretation:
- PID distributes mass across many non-null actions (including frequent strong action `10`), indicating persistent dosing activity rather than sparse corrective dosing.

## 2) TCU accounting fairness check

TCU mapping used for all controllers:
- `ACTION_VOLUMES_ML = [0, 5, 12, 30, 75, 180, 5, 12, 30, 75, 180]`
- `TCU = sum(ACTION_VOLUMES_ML[action_t])` over rollout timesteps.

Code path:
- Shared for every controller via `phase2_baselines_mechanism.rollout_metrics`.

Manual consistency check:
- Manual sum of per-action volumes and `rollout_metrics(...)[\"TCU\"]` were identical (difference `0.0`).

Conclusion:
- TCU accounting is identical and fair across controllers.

## 3) Is PID high TCU realistic behavior or projection artifact?

Observed in this diagnostic:
- PID has highest non-null-action fraction and much larger mean dose per timestep (`18.65 mL/step`) versus rule-based (`4.49`) and LUT (`4.48`).
- PID null condition is `|u| < 0.05`; otherwise continuous output is rounded into discrete bins (`1..5` or `6..10`).

Reasoning:
- In a continuous actuator system, small modulation around setpoint can be physically realistic.
- In this discrete projection, outputs that would be very small nonzero continuous adjustments get snapped to minimum non-null discrete doses repeatedly, increasing reagent usage.

Therefore:
- PID’s high TCU is **not a metric bug**.
- It is primarily a **control-policy behavior under discrete action projection** (i.e., a projection artifact/behavioral consequence of mapping continuous PID to coarse discrete doses), not unfair accounting.

## 4) Plain answer

- **Is TCU accounting fair and identical across controllers?** **Yes.**
- **Is PID high TCU genuine or artifactual?**  
  It is genuine for the implemented discrete-PID policy, and driven by the continuous-to-discrete projection behavior (frequent non-null dosing after quantization), not by a TCU computation error.

## Supporting sample means (same diagnostic run)

- Rule-based: `TCU_mean=2155.0`, `DCR_mean=85.0658`
- PID: `TCU_mean=8954.25`, `DCR_mean=96.4830`
- LUT: `TCU_mean=2152.5`, `DCR_mean=90.5405`
- Null: `TCU_mean=0.0`, `DCR_mean=41.6667`

Phase 3 was not started.
