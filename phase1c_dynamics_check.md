# Phase 1c Dynamics Check (Pre-Phase 2)

No parameters were changed in code. This file reports measured behavior only.

## 1) Action direction and magnitude with zero pH noise (`sigma_ph=0`)

Setup:
- fixed initial pH: 7.5
- each action 0-10 applied repeatedly for 6 steps
- environment otherwise unchanged

### Direction verification

- Actions 1-5 consistently **decrease** pH (acid side).
- Actions 6-10 consistently **increase** pH (alkaline side).
- Action 0 keeps pH unchanged in zero-noise mode.

### Measured per-step effect table

| Action | Direction | Mean ΔpH per step (approx) | |ΔpH| per step |
|---:|---|---:|---:|
| 0 | none | 0.000000 | 0.000000 |
| 1 | decrease | -0.006614 | 0.006614 |
| 2 | decrease | -0.015114 | 0.015114 |
| 3 | decrease | -0.033872 | 0.033872 |
| 4 | decrease | -0.069023 | 0.069023 |
| 5 | decrease | -0.123623 | 0.123623 |
| 6 | increase | +0.003504 | 0.003504 |
| 7 | increase | +0.008657 | 0.008657 |
| 8 | increase | +0.023494 | 0.023494 |
| 9 | increase | +0.076848 | 0.076848 |
| 10 | increase | +0.251427 | 0.251427 |

### Ordering check vs intended geometric progression

Within each sign branch, magnitude increases with action level:
- Acid branch: `1 < 2 < 3 < 4 < 5` (monotone in |ΔpH|)
- Alkaline branch: `6 < 7 < 8 < 9 < 10` (monotone in |ΔpH|)

So the dose-level ordering is preserved in transition effect size.

## 2) Noise vs signal

Configured:
- `sigma_ph` = **0.02**

Measured process-noise term (empirical std from 400 null-action steps):
- noise std = **0.021114773863457066**

Smallest non-null dose effect (from section 1):
- action **6**
- |mean ΔpH per step| = **0.003503599920918532**

Comparison:
- smallest-dose effect (**0.00350**) is **smaller** than 1σ noise (**0.02111**).

Plain finding:
- At current settings, the smallest non-null action effect is below one standard deviation of process noise.

## 3) Random vs always-null compliance, with start-state compliance fractions

Evaluation:
- 80 episodes each
- compliance metric: fraction of timesteps with pH in [6.5, 8.5]

| Policy | DCR mean | DCR SD | Start compliant frac | Start non-compliant frac |
|---|---:|---:|---:|---:|
| Random actions | 24.105901663912427 | 19.361166866387226 | 0.45 | 0.55 |
| Always null | 51.715176715176725 | 45.82157811910768 | 0.5625 | 0.4375 |

Interpretation for large null-policy variance:
- Start-state mix is split (about 56% compliant, 44% non-compliant), and always-null has no corrective action; this naturally creates wide episode-to-episode compliance spread, consistent with the high SD.

## Required flag (no adjustment applied)

The smallest dose effect is below 1σ process noise.  
Per instruction, this is flagged and no parameter adjustment is made here.
