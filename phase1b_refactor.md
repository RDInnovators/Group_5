# Phase 1b Refactor: Pure First-Principles Simulator Environment

## Objective

Refactor `WastewaterMDP` so pH transition is always computed by first-principles titration (`f_titration`), with no active LSTM/surrogate transition path.

## What changed (before -> after)

### 1) Transition source in `WastewaterMDP.step()`

- **Before:** hybrid branch
  - warm-start: `f_titration(...)`
  - post-warm: `self.lstm(...)` + inverse scaling for `dph`
- **After:** pure physics every step
  - always: `ph2_clean = f_titration(self.ph, action, self.A_T, self.C_T)`
  - process noise retained via `sigma_ph` stochastic update

### 2) Constructor decoupling from required surrogate inputs

- `lstm` type changed to `Optional[LSTMSurrogate]`.
- `dph_scaler` type changed to `Optional[StandardScaler]`.
- Environment now runs end-to-end with `lstm=None`.

### 3) Legacy LSTM transition retained-but-disabled

Per Phase 0 convention, legacy hybrid transition code is preserved as labeled comments in `step()` (not deleted), with explicit note that it is disabled for simulation-only Route A.

### 4) Legacy MC-dropout uncertainty penalty retained-but-disabled

The uncertainty penalty block that depended on surrogate predictive variance is preserved as labeled disabled comments (not executed in Route A).

## Verification outputs (executed)

## A) `lstm=None` runs 100+ steps (proof)

- Test: instantiate `WastewaterMDP(lstm=None, ...)` and run 150 random-action steps.
- Output: `LSTM_NONE_STEPS_OK 150`
- Result: no `TypeError`; environment runs with no surrogate present.

## B) Multi-step action -> pH trace (physical response + bounds)

Test setup:
- forced initial pH = 7.5 for clear directional response
- action sequence: `[1, 1, 1, 10, 10, 10, 0, 0]`
- pH sequence observed:
  - `[7.5, 7.514502, 7.542314, 7.510144, 7.668886, 7.842422, 8.12696, 8.119069, 8.12405]`
- bounds check: `TRACE_PH_BOUNDS_OK True`

Interpretation:
- pH remained within physical bounds [0,14].
- Acid/alkaline levels produced distinct trajectory changes under stochastic simulator noise.

## C) Random-policy vs always-null-policy compliance rates

Evaluation:
- fixed episode count: 40
- compliance metric: percent timesteps with pH in [6.5, 8.5]

Observed:
- Random policy: mean ± SD = **19.82699732699733 ± 16.573185734938274**
- Always-null policy: mean ± SD = **55.727650727650726 ± 45.45622020979086**

These are reported as measured, with no retuning.

## D) Phase 0 invariants re-check

- Observation dimension: `13`
- Action count: `11`
- Step signature: `(ndarray, float, bool, dict)`
- Compliance window: `6.5` to `8.5`
- Reward weights: `(2.0, -1.0, -0.3, -0.5, -0.1)`

All preserved.

## Status

Phase 1b completed for the environment refactor and required verification.

No retraining was started, and Phase 2 was not started.
