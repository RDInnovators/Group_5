# Manuscript Verification (Code + Result Artifacts)

Scope checked:
- `water_methodology_impl.py`
- `phase2_baselines_mechanism.py`
- `phase3_full_run.py`
- `train_one_job.py`
- `results/phase3_full/phase3b_evaluation_summary.json`
- `results/phase3_full/phase3c_stats.json`
- `actual_results.md`

---

## 1) Simulator stress test claim (92,730 cases, worst <1 ms)
**Status:** NOT-FOUND-IN-CODE  

- I did **not** find this exact stress-test result persisted in repository files (`*.py`, `*.md`, or result JSONs).
- There is no committed script/output file in the repo containing the exact `92,730`/`<1 ms` values.
- The titration functions exist and are used, but this specific benchmark output is not stored in the checked artifacts.

**Implication for manuscript:** if you cite this, either add the benchmark script/output artifact to the repo or rephrase as an internal diagnostic not in reproducibility files.

## 2) Newton-Raphson iteration cap
**Status:** CORRECT

- In `water_methodology_impl.py`, `solve_ph_newton_raphson` has:
  - `max_iter: int = 100`
  - loop: `for _ in range(max_iter):`
- `f_titration(...)` calls this solver.

## 3) Noise model (process 0.001 on transition, sensor 0.02 on observation)
**Status:** CORRECT

- Constants in `water_methodology_impl.py`:
  - `SIGMA_PH = 0.001` (process)
  - `SIGMA_PH_OBS = 0.02` (observation)
- Application:
  - Process noise applied in transition: `ph2 = ph2_clean + normal(0, sigma_process_ph)`.
  - Sensor noise applied in observation only: `_apply_obs_noise` perturbs observed pH channel with `sigma_obs_ph`.

## 4) State/action/reward claim
**Status:** PARTIALLY CORRECT (with important phase distinction)

- **State dim 13:** CORRECT (`TABLE8_COLS` length is 13 in `water_methodology_impl.py`).
- **Actions 11 discrete (null + 5 acid + 5 alkaline):** CORRECT (11 volumes in `ACTION_VOLUMES_ML`).
- **Reward weights for Phase 3:** CORRECT  
  - `W_COMP=3.0`, `W_DEV=-1.0`, `W_DOSE=-0.3`, `W_OVER=-0.5`, `W_ESC=-0.1`.
- **In-band penalty included in final Phase 3 run:** INCORRECT if stated as yes.  
  - Current code now has `W_INBAND_DOSE=-1.0` (Phase 4 fix experiment).
  - Phase 3 reported artifacts (`actual_results.md`, `phase3a_training_summary.json`) describe reward form as signed compliance with `W_COMP=3.0` and do **not** include this new term.
  - So manuscript text for **Phase 3 results** should state in-band penalty was **not used**.

## 5) PID tuning / deadband / anti-windup
**Status:** CORRECT

- `phase2_baselines_mechanism.py` documents:
  - Ziegler-Nichols method via `ziegler_nichols_pid_gains()`.
  - Deadband bounds in `PIDDiscreteControllerDeadband`: `deadband_lo=6.7`, `deadband_hi=8.3`.
  - Anti-windup behavior: integrator is held while idle in deadband (`return 0` path with integral hold comment/logic).

## 6) Compliance band pH 6.5–8.5
**Status:** CORRECT

- `PH_LO, PH_HI = 6.5, 8.5` in `water_methodology_impl.py`.
- Used throughout training/evaluation logic.

## 7) Evaluation counts + seeds + step budget
**Status:** CORRECT

- In `phase3_full_run.py`:
  - `SEEDS = (11, 22, 33)`
  - `LOCKED_PPO_TOTAL_STEPS = 5_000_000`
  - `LOCKED_DDPG_TOTAL_STEPS = 5_000_000`
  - `TIER1_EPISODES = 500`
  - `TIER2_EPISODES = 200`

## 8) Tier 2 OOD definition (actual parameters and ranges)
**Status:** PARTIALLY CORRECT (concept right; manuscript needs specific numeric detail)

- Actual shift implementation in `phase3_full_run.py`:
  - Tier 2 samples **A_T** and **C_T** per episode.
  - `A_T` sampled uniformly from `A_T_lo..A_T_hi`.
  - `C_T` sampled uniformly from `C_T_lo..C_T_hi`.
- Actual computed ranges (from `_derive_tier2_shift_ranges()` on current data):
  - `A_T_lo = 0.5`
  - `A_T_hi = 4.768534000000001`
  - `C_T_lo = 0.5`
  - `C_T_hi = 0.5`
  - Source string: `derived from real DS-2 quantiles (hardness_mgL, conductivity_uScm)`

**Important nuance:** with these computed values, `C_T` is effectively fixed at 0.5 (not varied), while `A_T` varies.

## 9) CER formula
**Status:** CORRECT

- In `phase3_full_run.py` metric computation:
  - `CER = DCR / (1.0 + TCU)`
- This matches manuscript claim.

## 10) Reward scaling training-only, not metrics
**Status:** CORRECT

- Training:
  - PPO uses `rew_buf.append(r * LOCKED_REWARD_SCALE)` with `LOCKED_REWARD_SCALE=0.1`.
  - DDPG buffer stores `r * LOCKED_REWARD_SCALE`.
- Evaluation metrics:
  - Computed from trajectory states/actions (`DCR`, `TCU`, `CER`, etc.) in `_episode_metrics`, independent of training reward scaling.
- `actual_results.md` explicitly states this fairness property.

## 11) DDPG implementation and parity budget
**Status:** CORRECT

- DDPG is continuous-actor with projection to discrete action index:
  - `wes.DDPGDiscrete.u_to_action(...)`.
- Trained on same seeds and 5M budget in `phase3_full_run.py`:
  - seeds `(11,22,33)`, total steps `5,000,000`.

---

## Mismatches to fix in manuscript text

1. **In-band penalty in Phase 3 results**: should be stated as **not used** in reported Phase 3 numbers (it is a later Phase 4 experiment).
2. **Tier 2 shift details**: include exact implemented ranges and note that `C_T` was effectively fixed at 0.5 in the executed run.
3. **Stress-test benchmark citation**: not currently reproducible from committed artifacts; add artifact/script output or soften claim.
