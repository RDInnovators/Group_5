# Phase 2: Baselines + Progressive-Dosing Mechanism (No PPO training)

This phase implemented comparator policies and mechanism toggles, then ran short unit/sanity checks only.

No PPO training was started.

## Phase 2a — Baseline controllers implemented

Implemented in `phase2_baselines_mechanism.py`:

1. **Rule-based threshold controller**
   - Logic:
     - if `pH < 6.5` -> alkaline dose action `8`
     - if `pH > 8.5` -> acid dose action `3`
     - else -> action `0`

2. **PID controller (Ziegler–Nichols)**
   - Method: documented closed-loop Ziegler–Nichols form using fixed characterized pair (`Ku=4.0`, `Pu=2.5`).
   - Gains:
     - `Kp = 0.6*Ku = 2.4`
     - `Ki = 2*Kp/Pu = 1.92`
     - `Kd = Kp*Pu/8 = 0.75`
   - Continuous PID output mapped to nearest discrete dose action (11-action set).

3. **Static lookup-table controller**
   - Precomputed from pH-deviation bins around target `PH_MID=7.5`:
     - `e<=-1.0 -> 10`
     - `-1.0<e<=-0.6 -> 9`
     - `-0.6<e<=-0.3 -> 8`
     - `-0.3<e<=-0.1 -> 7`
     - `-0.1<e<0.1 -> 0`
     - `0.1<=e<0.3 -> 2`
     - `0.3<=e<0.6 -> 3`
     - `0.6<=e<1.0 -> 4`
     - `e>=1.0 -> 5`

4. **DDPG (continuous projected to discrete actions)**
   - Implemented actor-critic (`DDPGDiscrete`) with projection `u∈[-1,1] -> action∈{0..10}`.
   - Phase-2 run-check executed (not Phase-3 training): 1000 update steps completed.

### Short sanity evaluation (non-learning baselines, 40 episodes)

Measured DCR mean±SD (all numbers from executed run):

- Rule-based: **90.42099792099792 ± 14.919824016275543**
- PID: **96.25779625779626 ± 3.5338331405389827**
- LUT: **7.978045635548779 ± 22.115712640256838**

Reference policies in same run:
- Random: **29.61018711018711 ± 14.101333844969544**
- Always-null: **50.0 ± 50.63696835418333**

Plain note: LUT underperforms both random and null in this sanity run.

## Phase 2b — Progressive low-to-high mechanism

Implemented as independently toggleable controls in `WastewaterMDP`:

- `enable_curriculum_masking: bool = True`
- `enable_escalation_penalty: bool = True`

### Structural reward incentive

Reward already contains:
- quadratic dose penalty term (`W_DOSE * Rdo`) with `W_DOSE=-0.3`
- escalation penalty term (`W_ESC * Re`) with `W_ESC=-0.1`

These encode standing preference for lower-dose/smoother dosing.
Escalation penalty can now be turned off independently via `enable_escalation_penalty=False`.

### Curriculum masking schedule

Masking behavior:
- During warm-up (`global_step < curriculum_steps`), high actions `{4,5,9,10}` are stochastically masked with probability increasing as pH deviation decreases.
- After warm-up, all actions are unmasked.
- Entire mechanism can be disabled via `enable_curriculum_masking=False`.

### Unit check: masking restriction and release

Observed masks from executed run:
- Warm-up mask allowed actions: **7/11**
- Post-warm mask allowed actions: **11/11**

### Unit check: escalation toggle effect

Same aligned state/action, toggling only escalation penalty:
- raw reward (with penalty): `2.9999999999999996`
- raw reward (without penalty): `2.8999999999999995`
- difference (without - with): `-0.10000000000000009`

This confirms the toggle changes the reward contribution for escalation events.

## Phase 2c — Metric instrumentation

Rollout metric computation implemented in `rollout_metrics(...)`:
- `DCR`
- `MPD` (mean pH deviation)
- `TCU` (total chemical usage)
- `CER = (DCR/TCU)*1000` (set to NaN when `TCU=0`)
- `OEC` (overshoot event count)
- `PST` (pH stabilization time)
- `PDCR` (progressive-dosing compliance ratio)
- `STG` flagged as NaN (not applicable / future work in Route A)

Single short instrumented rollout (executed) produced:
- `DCR=100.0`
- `MPD=0.0920990121996083`
- `TCU=0.0`
- `CER=NaN`
- `OEC=0.0`
- `PST=7.0`
- `PDCR=1.0`
- `STG=NaN`

## Phase 2 status

Phase 2 implementation/unit checks completed.

No PPO training was started, and Phase 3 was not started.
