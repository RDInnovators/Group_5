# Progress Report

## 1. Project Identity

- **Project name:** `Water - MBRL wastewater dosing (Methodology v.01)` (from `README.md`).
- **Research topic:** model-based reinforcement learning (MBRL) for adaptive pH/chemical dosing in industrial wastewater neutralization, combining physics-informed simulation, action reconstruction, LSTM surrogate modeling, and PPO control.
- **Target venue/journal (identifiable):** documentation repeatedly references **Journal of Water Process Engineering-style** framing; no formal submission metadata file is present.

## 2. Completed Components

### Code and Pipeline Modules

- `water_methodology_impl.py` implements the core end-to-end methodology components (physics titration model, preprocessing, action reconstruction, LSTM surrogate, MDP, PPO training, evaluation helpers, and orchestrated pipeline run entrypoint).
- `water_rdi_loaders.py` implements Table-2 dataset loading/mixing across NWIS, bundled CSVs, and KU-MWQ/NWIS DS-5 paths, with synthetic fallback controls and strict "require real" mode.
- `water_experiments_small.py` implements reduced-budget evaluation utilities (Sec 2.3.2 diagnostic, Table-6 gate checks, baseline comparisons including LUT/DDPG, and consolidated smoke-run aggregation).
- `paper_figures.py` implements figure-generation utilities and saves publication-style PNG outputs to `figures_paper/`.
- `scripts/paper_manifest.py` implements reproducibility manifest reporting (Python/package versions, git hash, and optional Table-2 loader probe).

### Notebook and Documentation

- `methodology_implementation.ipynb` contains a complete staged walkthrough (data assembly to PPO and evaluation), with executed outputs and saved figures.
- `README.md` provides project identity, quick start, and document map.
- `Methodology_v01_Walkthrough.md` provides a structured methodology narrative mapped to manuscript sections.
- `Implementation_and_Remaining_Work.md` provides explicit implemented/partial/not-implemented scope mapping against the manuscript.
- `Results_Analysis_Figures.md` provides figure-by-figure interpretation guidance and reporting checklist.

### Data and Artifacts Present

- `data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv` exists as a DS-2 paired proxy dataset.
- `data/rdi/ds3_wqp_effluent_md_proxy.csv` exists as a DS-3 effluent/compliance proxy dataset.
- `data/rdi/ds2_spc_MD_sample.csv` exists as an additional DS-2 sample dataset.
- `data/rdi/KU-MWQ A Dataset for Monitoring Water Quality Using Digital Sensors/Sensor data for 30 cm.xlsx` exists (binary Excel file; not directly readable by the text reader).
- `figures_paper/fig_ds1_ph_timeseries.png` exists and is referenced by docs/notebook as DS-1 time-series output.
- `figures_paper/fig_ds5_sensors.png` exists and is referenced by docs/notebook as DS-5 multi-sensor output.
- `figures_paper/fig_ds4_monthly_ph.png` exists and is referenced by docs/notebook as DS-4 monthly/OOD output.
- `figures_paper/fig_dcr_boxplot.png` exists and is referenced by docs/notebook as Tier-1 DCR comparison output.
- `figures_paper/fig_ood_feature_shift.png` exists and is referenced by docs/notebook as OOD shift output.
- `requirements_methodology.txt` defines installable methodology dependencies.
- `requirements-paper-lock.txt` provides pinned paper-run dependency versions.
- `.env.example` provides environment-variable configuration template.
- `.vscode/settings.json` provides project interpreter path configuration.
- `.gitignore` is configured with environment, cache, and local file exclusions.

## 3. In-Progress Components

- `scripts/writer_preflight.py` is partially complete in intent, but currently unusable due to a syntax/indentation bug (`IndentationError` after `try:`), so preflight validation cannot run.
- `methodology_implementation.ipynb` is operational but configured for **demo-scale** defaults (`DEMO_MODE=True` in notebook cells), so manuscript-scale budgets and long-run experiments are not yet the default executed state.
- `water_experiments_small.py` implements "small/first-pass" experiments and explicitly does not implement the full manuscript-scale protocol (reduced episodes/steps and smoke-testing orientation).
- `water_methodology_impl.py` implements only a subset of manuscript acceptance gates in automated form (Table-6 subset), while other gates remain outside current automation per project docs.
- `Results_Analysis_Figures.md` is interpretive guidance and explicitly states that numeric examples depend on the latest notebook run (not a fixed finalized result table).

## 4. Not Yet Started (Planned but No Corresponding Full Implementation File)

Based on `Implementation_and_Remaining_Work.md` and `Methodology_v01_Walkthrough.md`, the following planned items are called out but do not have dedicated full implementation artifacts in this repo:

- Full Sec 2.3.2 manuscript-scale validation protocol (50x120-minute open-loop trajectory study) as a dedicated study script/report artifact.
- Full Tier-2/Tier-3 protocol runs as dedicated separate large-run experiment pipelines.
- Stable-Baselines3/Gymnasium implementation path (current code uses custom environment and custom PPO).
- MLflow/joblib pipeline serialization workflow for experiment tracking and packaging.
- Full ablation grid execution (Table 17) with full seed sweeps.
- Hyperparameter sensitivity sweep framework (Sec 13-level coverage).
- Full multi-architecture benchmark scripts for GRU/TCN/MLP (Table 7 mention only).
- Extended statistical supplement implementation (bootstrap CIs, Holm-Bonferroni, etc.) beyond current Wilcoxon helper coverage.

## 5. Key Results So Far

Numeric outputs are present in executed notebook cells (`methodology_implementation.ipynb`):

- **Table-2 loader flags:** all true in the shown run (`DS-1`..`DS-5` real/public/bundled paths used).
- **DS-3 compliance positive fraction:** `0.854125`.
- **Estimated buffering proxies:** `A_T = 1.2658695`, `C_T = 0.5`.
- **CUSUM/inverse reconstruction:** non-null action fraction `0.0`, CUSUM event rate `0.0`.
- **Table-3 diagnostics output:** `n_timesteps = 5750`, `n_cusum_windows = 0`, `discard_rate = 0`, `null_action_fraction = 1.0`, `inverse_residual_mae = NaN`, `cusum_flag_timestep_rate = 0`, `cusum_fp_proxy_flat_noise = 0`.
- **LSTM training output:** `sigma_model (delta-pH RMSE) = 0.015589298203192064`.
- **Sec 2.3.2 diagnostic output:** `MAPE_pct = 88.69296059866576`, `MAPE_pct_median = 100.0`, `median_abs_dph_err = 0.07732879170625484`, `n = 80`.
- **LSTM validation metrics:** `surrogate_val_RMSE_pH = 0.015726209628388266`, `surrogate_MAE_dph_residual = 0.00602950025005767`, `n_residuals = 814`.
- **Table-6 gate status (shown run):** all three implemented gates pass (`True`, `True`, `True`) against thresholds (`0.1`, `0.07`, `0.25`).
- **DS-5 LSTM proxy metrics:** `median_abs_dph_err = 0.05036475509405136`, `rmse_dph = 0.0629935079446479`, `n = 622`.
- **MC-dropout calibration:** `uncertainty p95 = 0.0031234248541295523`.
- **Tier-1 DCR means:** PPO `13.210323119639428`, RBT `91.63201663201664`, PID `98.44074844074844`, LUT `99.49757449757449`.
- **Wilcoxon tests (PPO vs baselines, Bonferroni alpha/3):** all shown `p = 0.00048828125` with `stat = 0.0`.
- **OOD feature shift examples (DS-4 minus train, normalized):** includes `pH_raw = 1.6883330414101738`, `cond = 4.442733343437176`, etc.

## 6. Blocking Issues

- `scripts/writer_preflight.py` currently fails immediately with `IndentationError` (line after `try:`), blocking the intended preflight workflow.
- `data/rdi/KU-MWQ ... /Sensor data for 30 cm.xlsx` is a binary Excel file and not directly inspectable with the text reader; validation depends on runtime Excel loading (`openpyxl` path).
- Real-data execution paths depend on external NWIS availability/connectivity; code contains explicit fallbacks to synthetic data when unavailable.
- Multiple code paths require optional dependency availability (`openpyxl`, scientific stack); missing packages will break corresponding imports/loaders (as indicated by explicit import error handling in source).

## 7. Immediate Next Steps

- Fix `scripts/writer_preflight.py` indentation bug and rerun preflight to restore baseline health checks.
- Run the notebook/pipeline with `DEMO_MODE=False` for manuscript-scale training/evaluation budgets and regenerate metrics/figures.
- Implement remaining acceptance gates and full Sec 2.3.2 long-window validation as dedicated reproducible scripts.
- Add dedicated Tier-2/Tier-3 large-run experiment scripts plus persisted results summaries (not only notebook output).
- Expand statistical reporting and architecture baselines (GRU/TCN/MLP + broader multiple-comparison/statistical supplements) to close documented manuscript gaps.
