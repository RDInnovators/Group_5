# Phase 0 Reconciliation (Simulation-Only Route A)

## Scope

Goal of this phase: reconcile the codebase from the prior real-data/surrogate framing to the new simulation-only pH-dosing framing, **without deleting legacy code**.

## Important input-note

Requested reference documents `Literature_Review.docx` and `Research_Questions.docx` were not found in the repository by filename. Reconciliation in this phase therefore used:
- the user-provided framing in chat,
- current code structure,
- existing manuscript/doc artifacts in repo.

## What was disabled (and where)

### 1) Real-data training path + CUSUM reconstruction path + LSTM-on-reconstructed-actions path

- **File:** `water_methodology_impl.py`
- **Function:** `reconstruct_actions(...)`
- **Before:** executed CUSUM windowing + inverse assignment to generate action labels from real pH series.
- **After:** legacy implementation retained as comments; function now raises a labeled `RuntimeError` stating this path is disabled for simulation-only Route A.

- **File:** `water_methodology_impl.py`
- **Function:** `run_full_pipeline(...)`
- **Before:** orchestrated legacy end-to-end flow:
  real/bundled data loading -> reconstructed actions -> preprocessing -> LSTM training -> hybrid env PPO -> evaluation.
- **After:** function now immediately raises a labeled `RuntimeError`; legacy implementation remains in place below the guard for traceability.

### 2) Real-data Tier-3 evaluation helper path

- **File:** `water_experiments_small.py`
- **Function:** `methodology_first_pass_small(...)`
- **Before:** invoked legacy `run_full_pipeline` and included Tier-3 sim-to-real note path tied to DS-5.
- **After:** function now immediately raises a labeled `RuntimeError` indicating the helper is disabled for simulation-only Route A; legacy implementation is retained below.

### 3) Auxiliary real-data probe in manifest script

- **File:** `scripts/paper_manifest.py`
- **Before:** optional `build_table2_mixed(...)` probing printed DS-1/DS-5 table flags.
- **After:** replaced with a labeled message that real-data table probing is disabled for simulation-only Route A.

## What remains intact (required by Route A)

- First-principles titration core remains intact in `water_methodology_impl.py`:
  - `alkalinity_from_ph`
  - `solve_ph_newton_raphson`
  - `f_titration`
  - action-dose mapping via `ACTION_VOLUMES_ML`
- MDP/RL structural components still present:
  - `WastewaterMDP`
  - `ppo_train`, `ActorCritic`
  - baseline controllers (`rule_based_action`, `PIDController`)

## Standalone simulator environment check

A direct smoke test instantiated `WastewaterMDP` with minimal placeholder model/scaler objects (no real-data path) and executed `reset` + `step`.

Observed interface:
- Observation dimension: **13**
- Action count: **11** (null + 5 acid + 5 alkaline via `ACTION_VOLUMES_ML`)
- `step` returns: `(ndarray, float, bool, dict)`
- Compliance window constants: `PH_LO=6.5`, `PH_HI=8.5`
- Reward weights: `W_COMP=2.0`, `W_DEV=-1.0`, `W_DOSE=-0.3`, `W_OVER=-0.5`, `W_ESC=-0.1`

## Major remaining mismatches vs simulation-only Route A

These are **not** fixed in Phase 0; they should be addressed in subsequent phases:

1. `WastewaterMDP` currently still requires LSTM-related constructor inputs and supports hybrid dynamics by design; it is not yet refactored into a pure simulator-only transition path.
2. Several legacy symbols/docs/constants still reference Table-2/real-data framing.
3. Evaluation helpers in `water_experiments_small.py` still contain legacy code bodies below the new runtime guard (retained intentionally), and a new Route A-specific experiment entrypoint is not yet created.

## Phase 0 status

Phase 0 objective (disable legacy real-data/CUSUM/LSTM-reconstruction/Tier-3 paths without deleting code) is completed.

Because major architectural mismatch (hybrid-env constructor coupling to LSTM objects) remains, Phase 1 should proceed by introducing a simulator-only environment entrypath while preserving documented MDP structure.
