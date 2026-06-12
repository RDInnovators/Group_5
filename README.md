# RL vs Classical Control for pH Neutralization

## 1. Project Overview

This repository implements a physics-grounded benchmark for compliance-oriented pH neutralization control in industrial wastewater simulation, comparing reinforcement learning (PPO/DDPG variants) against classical controllers (rule-based, PID with deadband, LUT). The reported Phase 3 results are simulation-only and are summarized in `actual_results.md`, with manuscript text in [`Manuscript_Publishable.docx`](Manuscript_Publishable.docx). The headline result is that classical control, especially `pid_deadband`, outperforms `ppo_full` on Tier-1 compliance and chemical efficiency metrics.

## 2. Repository Structure

```text
.
├── actual_results.md                          # Final Phase 3 metrics, hypothesis verdicts, and statistics summary
├── Manuscript_Publishable.docx               # Current manuscript draft (publishable write-up target)
├── manuscript_verification.md                # Claim-by-claim manuscript fact check against code/results
├── requirements_methodology.txt              # Main Python dependencies for simulator/training workflow
├── requirements-paper-lock.txt               # Locked dependencies for paper/figure reproducibility
├── water_methodology_impl.py                 # Core simulator, MDP, reward, PPO model, and physics primitives
├── water_experiments_small.py                # Small-scale experiment utilities and DDPG components
├── water_rdi_loaders.py                      # Dataset loading/mixing utilities for Table-2 data roles
├── phase2_baselines_mechanism.py             # Baseline controllers and baseline-oriented env helpers
├── phase3a_train.py                          # Phase 3a seed training script (PPO + DDPG artifacts)
├── phase3_full_run.py                        # Full Phase 3 orchestrator (training, eval, stats, report)
├── train_one_job.py                          # Subprocess-isolated single-job trainer used by Phase 3
├── phase4_fix1_run.py                        # Phase 4 fix1 retrain/eval orchestrator
├── train_one_job_fix1.py                     # Subprocess trainer for Phase 4 fix1 PPO run
├── paper_figures.py                          # Figure-generation script for paper plots
├── methodology_implementation.ipynb          # Notebook walkthrough / implementation notebook
├── data/
│   ├── rdi/                                  # Small/medium RDI CSV inputs used by loaders/analysis
│   └── n2o_dataset/                          # N2O validation dataset files (large CSVs gitignored)
├── results/
│   ├── phase3_full/                          # Main reported run: curves, models, eval CSVs, summary JSONs
│   │   ├── curves/                           # Per-seed training curves by controller/variant
│   │   ├── eval/                             # Tier-1/Tier-2 episode-level evaluation outputs
│   │   ├── models/                           # Saved PPO/DDPG model checkpoints
│   │   ├── phase3a_training_summary.json     # Locked training provenance
│   │   ├── phase3b_evaluation_summary.json   # Aggregated Tier-1/Tier-2 metric summaries
│   │   └── phase3c_stats.json                # Wilcoxon/Bonferroni/bootstrap statistics
│   ├── phase3a/                              # Earlier Phase 3a intermediate artifacts
│   ├── phase3a_singlecheck/                  # Diagnostic probe artifacts (critic/optimization checks)
│   ├── phase4_failure_diagnosis/             # PPO underperformance diagnosis outputs
│   └── phase4_fix1/                          # In-band-penalty retrain artifacts
├── figures_paper/                            # Generated figures used in reports/manuscript
└── scripts/
    ├── paper_manifest.py                     # Manifest/check tooling for paper assets
    └── writer_preflight.py                   # Preflight utility for writing/report workflows
```

## 3. Key Files Index

| Purpose | Path | Description |
|---|---|---|
| Titration simulator + MDP | `water_methodology_impl.py` | First-principles titration (`f_titration`), `WastewaterMDP`, reward terms/constants, PPO model class. |
| PPO/DDPG training components | `water_methodology_impl.py`, `water_experiments_small.py` | PPO architecture/training helpers and DDPG network/utility components used by run scripts. |
| Baseline controllers | `phase2_baselines_mechanism.py` | Rule-based, deadband PID, LUT logic and simulator env wrappers. |
| Phase 3 full orchestrator | `phase3_full_run.py` | Locked-config training orchestration, evaluation, statistics, and `actual_results.md` generation. |
| Single training subprocess entry | `train_one_job.py` | One `(variant, seed)` training job in isolated process (Phase 3 reliability fix). |
| Phase 4 fix1 orchestrator | `phase4_fix1_run.py` | Retraining/evaluation workflow for in-band dosing penalty experiment. |
| Phase 4 fix1 single-job entry | `train_one_job_fix1.py` | Subprocess entry for Phase 4 fix1 job execution. |
| Main results narrative | `actual_results.md` | Headline metrics, comparisons, and H1-H4 verdicts. |
| Tier-1/Tier-2 summary JSON | `results/phase3_full/phase3b_evaluation_summary.json` | Controller-by-controller aggregate metric means/SDs. |
| Statistical tests JSON | `results/phase3_full/phase3c_stats.json` | Wilcoxon tests, Bonferroni correction, effect sizes, bootstrap CIs. |
| Training provenance JSON | `results/phase3_full/phase3a_training_summary.json` | Locked config, seeds, artifacts, and run metadata. |
| Manuscript draft | `Manuscript_Publishable.docx` | Draft manuscript text aligned to current repo results. |
| Manuscript verification report | `manuscript_verification.md` | Verification of manuscript claims against concrete code/results. |
| Phase 0 report | `phase0_reconciliation.md` | Route-A reconciliation to simulation-only study framing. |
| Phase 1a report | `phase1a_transition_verification.md` | Pre-refactor transition behavior verification. |
| Phase 1b report | `phase1b_refactor.md` | Refactor to pure first-principles environment transitions. |
| Phase 1c report | `phase1c_dynamics_check.md` | Dynamics/action-direction checks before Phase 2. |
| Phase 1d report | `phase1d_noise_model.md` | Process-vs-observation noise model correction. |
| Phase 2 report | `phase2_baselines_mechanism.md` | Baselines and progressive-dosing mechanism implementation summary. |
| Phase 2 fixes | `phase2fix.md`, `phase2fix2.md`, `phase2fix3.md`, `phase2fix4.md` | Iterative fixes for LUT/CER/TCU/deadband fairness details. |
| Phase 3 reward diagnosis/fix reports | `phase3_reward_diagnosis.md`, `phase3_reward_fix.md`, `phase3_reward_align.md` | Reward-function failure diagnosis and compliance-aligned reward locking. |
| Phase 3 dynamics/probe reports | `phase3_learning_diagnosis.md`, `phase3_optimization_diagnosis.md`, `phase3_critic_fix.md`, `phase3_probeA_extended.md`, `phase3_singleseed_check.md`, `phase3a_interim_seed11.md` | PPO learning/critic diagnostics and probe outcomes. |
| Phase 4 reports | `phase4_failure_diagnosis.md`, `phase4_fix1_inband_penalty.md` | PPO underperformance diagnosis and fix1 intervention notes. |

## 4. Results Summary

Tier-1 headline metrics below are taken from `results/phase3_full/phase3b_evaluation_summary.json` (matching `actual_results.md`):

| Controller | DCR (Tier 1, %) | TCU (Tier 1) | CER (Tier 1) |
|---|---:|---:|---:|
| `rule_based` | 89.54 | 1508.88 | 48.7189 |
| `pid_deadband` | 98.24 | 1873.44 | 38.6938 |
| `lut` | 94.26 | 1509.12 | 48.7204 |
| `ppo_full` | 71.03 | 20671.92 | 4.1555 |
| `ddpg` | 36.50 | 4065.96 | 32.8676 |
| `null` | 49.30 | 0.00 | 49.3010 |
| `random` | 27.45 | 26314.48 | 0.0010 |

Hypothesis verdicts (from `actual_results.md`): **H1 not supported; H2 not supported; H3 supported; H4 deferred**.

Full detail files:
- `actual_results.md`
- `results/phase3_full/phase3b_evaluation_summary.json`
- `results/phase3_full/phase3c_stats.json`
- `results/phase3_full/eval/`

## 5. Documentation / Report Index

All report-style Markdown files at repo root:

| Report | Description |
|---|---|
| `progress_report.md` | Consolidated project progress log across phases and pivots. |
| `n2o_validation_check.md` | Assessment of N2O dataset suitability for simulator validation. |
| `simulator_revalidation.md` | Revalidation checks for simulator behavior before later phases. |
| `phase0_reconciliation.md` | Reconciles codebase to simulation-only Route-A framing. |
| `phase1a_transition_verification.md` | Verifies transition behavior before first-principles refactor. |
| `phase1b_refactor.md` | Documents pure first-principles environment refactor. |
| `phase1c_dynamics_check.md` | Confirms action direction/magnitude and dynamics sanity. |
| `phase1d_noise_model.md` | Splits process and observation noise modeling. |
| `phase2_baselines_mechanism.md` | Phase 2 implementation of baselines and progressive mechanism. |
| `phase2fix.md` | First Phase 2 fix notes prior to Phase 3 launch. |
| `phase2fix2.md` | CER redefinition update and rationale. |
| `phase2fix3.md` | Diagnostic analysis of PID/TCU behavior. |
| `phase2fix4.md` | Deadband PID fairness correction details. |
| `phase3a_interim_seed11.md` | Interim single-seed PPO health snapshot during Phase 3a. |
| `phase3_reward_diagnosis.md` | Diagnosis of PPO reward-function failure modes. |
| `phase3_reward_fix.md` | Reward normalization/compliance fix decisions. |
| `phase3_reward_align.md` | Compliance-aligned reward locking before full run. |
| `phase3_learning_diagnosis.md` | PPO learning dynamics diagnosis (stability/variance issues). |
| `phase3_optimization_diagnosis.md` | Optimization-level PPO diagnosis and checks. |
| `phase3_critic_fix.md` | Critic learnability probes and selected fix candidates. |
| `phase3_probeA_extended.md` | Extended Probe-A run results and interpretation. |
| `phase3_singleseed_check.md` | Additional single-seed PPO locked-config check. |
| `phase4_failure_diagnosis.md` | Post-Phase-3 diagnosis of PPO underperformance. |
| `phase4_fix1_inband_penalty.md` | Fix #1 (in-band dosing penalty) design and outcomes. |
| `manuscript_verification.md` | Verification of manuscript claims vs actual repo artifacts. |
| `actual_results.md` | Final Phase 3 result summary and hypothesis outcomes. |
| `Methodology_v01_Walkthrough.md` | Walkthrough of methodology components and mapping to code. |
| `Implementation_and_Remaining_Work.md` | Implemented scope vs remaining work toward publication. |
| `Results_Analysis_Figures.md` | Figure-by-figure interpretation guide for `figures_paper/`. |

## 6. Configuration

Locked final configuration used for the reported full run (verified from `phase3_full_run.py` and `water_methodology_impl.py`):

- `reward_scale = 0.1` (training updates only)
- `gamma = 0.99`
- `vf_coef = 0.5`
- advantage normalization: **on**
- signed reward with compliance term and weights:
  - `W_COMP = 3.0`
  - `W_DEV = -1.0`
  - `W_DOSE = -0.3`
  - `W_OVER = -0.5`
  - `W_ESC = -0.1`
- compliance band: `pH 6.5-8.5`
- action space: `11` discrete actions (`0..10`)
- state/observation dimension: `13`
- seeds: `11, 22, 33`
- training budget: `5,000,000` steps (PPO and DDPG)
- evaluation episodes: `Tier1=500`, `Tier2=200`

## 7. How to Reproduce

1. Create environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements_methodology.txt
   ```
2. Run the full benchmark pipeline:
   ```bash
   python3 phase3_full_run.py
   ```
3. Find outputs under `results/phase3_full/`:
   - models: `results/phase3_full/models/`
   - learning curves: `results/phase3_full/curves/`
   - eval episodes/aggregates: `results/phase3_full/eval/`
   - summaries/statistics: `results/phase3_full/phase3a_training_summary.json`, `results/phase3_full/phase3b_evaluation_summary.json`, `results/phase3_full/phase3c_stats.json`
4. Regenerate gitignored large artifacts by re-running training/data generation:
   - DDPG large curves (e.g., `results/phase3_full/curves/ddpg/ddpg_seed_33_curve.csv`)
   - large N2O CSVs in `data/n2o_dataset/`

## 8. Notes / Known Issues

- The study is currently simulation-only; no public real dataset in this repo was accepted as suitable for direct real-trajectory validation (see `n2o_validation_check.md` and `actual_results.md` threats-to-validity section).
- The ~265MB DDPG curve artifacts are treated as logging artifacts and are gitignored (`results/phase3_full/curves/ddpg/ddpg_seed_11_curve.csv`, `results/phase3_full/curves/ddpg/ddpg_seed_22_curve.csv`, `results/phase3_full/curves/ddpg/ddpg_seed_33_curve.csv`).
- DOI/reference metadata in manuscript references should be verified before submission (manuscript file: `Manuscript_Publishable.docx`).
