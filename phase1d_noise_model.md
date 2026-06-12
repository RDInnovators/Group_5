# Phase 1d Noise Model Correction (Process vs Observation)

This phase applies a **modeling correction** (not results tuning): separate process noise on true pH transition from sensor noise on observed pH.

## What changed

## 1) Process noise on true transition

- **Before:** single `sigma_ph=0.02` added directly to true transition (`ph2 = ph2_clean + N(0, sigma_ph)`).
- **After:** process transition noise set to small value:
  - `SIGMA_PH = 0.001` (used as process-noise std on true pH transition)

Physical rationale:
- True neutralization state should not randomly jump at sensor-scale every step without actuation.
- Small unresolved disturbances/noise still modeled in transition, but below low-dose actuation scale.

## 2) Observation/sensor noise on observed pH only

- Added separate observation-noise constant:
  - `SIGMA_PH_OBS = 0.02`
- Applied in `_apply_obs_noise(...)` to observed pH channel only (observation vector), not to true state.

Interpretation:
- Agent sees noisy pH readings at sensor-scale uncertainty.
- Underlying process dynamics remain governed by `f_titration(...)` + small process noise.

## 3) Legacy single-noise logic retained-but-disabled

- Legacy single-sigma transition noise and legacy observation augmentation logic are preserved as labeled disabled comments (Phase 0 convention), not deleted.

## Re-verification (no training)

## 3) Action direction/ordering at zero process noise

Re-ran action table with `sigma_process=0`, `sigma_obs=0` from fixed pH 7.5.

- Actions 1-5: consistent pH decrease
- Actions 6-10: consistent pH increase
- Ordering preserved (small dose -> small |ΔpH|, larger dose -> larger |ΔpH|)

Smallest non-null mean per-step effect:
- action 6: `|ΔpH| = 0.003503599920918532`

## 4) Process-noise std vs smallest-dose effect

Configured process noise:
- `sigma_process = 0.001`

Empirical process-noise std (measured from null-action steps):
- `0.000959903836612872`

Smallest non-null dose effect:
- `0.003503599920918532`

Ratio:
- `0.0035036 / 0.001 = 3.5036x`

3x criterion:
- **met** (`meets_3x_rule = true`)

## 5) Observation noise presence vs true-state cleanliness

Test setup:
- `sigma_process=0`
- `sigma_obs=0.02`
- null action rollout

Measured:
- std(observed_pH - true_pH) = `0.019408447340395086` (visible ~0.02 scale)
- std(true_transition - clean_f_titration_transition) = `0.0`

Conclusion:
- Observation noise is present at expected scale.
- True transition remains clean when process noise is zero.

## 6) Random vs always-null compliance sanity (80 episodes)

| Policy | DCR mean | DCR SD | Start compliant frac | Start non-compliant frac |
|---|---:|---:|---:|---:|
| Random | 29.095634095634097 | 16.470866576899894 | 0.475 | 0.525 |
| Always null | 44.812889812889814 | 49.62169081094322 | 0.4625 | 0.5375 |

Sanity-only reporting; no target fitting performed.

## Status

Phase 1d complete. No training started; Phase 2 not started.
