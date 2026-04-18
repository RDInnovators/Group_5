# Implementation status - what is built, what is partial, what is left

This file replaces the old scattered status / alignment notes. It describes the **`Water/`** codebase relative to *Methodology v.01* (`Methodology v.01.docx` - keep your copy **locally**; it is not distributed with the repo).

---

## Fully implemented (core pipeline)

| Manuscript area | Code / notebook |
|-----------------|-----------------|
| **Table 2 - mixed public data** | `water_rdi_loaders.build_table2_mixed`: NWIS IV/DV where reachable, bundled WQP CSVs for DS-2/DS-3, **DS-5** from KU-MWQ workbook (`WATER_DS5_SOURCE=auto`) or NWIS IV |
| **\(f_{\mathrm{titration}}\)** - titration step | `water_methodology_impl.f_titration`, alkalinity / \(A_T, C_T\) from DS-2 + DS-1 |
| **CUSUM + inverse assignment** | `cusum_events`, `reconstruct_actions`, `assign_action_inverse`, Table 3-style metrics |
| **Preprocessing P1-P6** | `preprocess_monitor`: MinMax on inputs, StandardScaler on **ΔpH** for LSTM, rolling features, chronological split |
| **Finite features for ML** | Extra guard so MinMax inputs do not contain NaNs (stabilizes LSTM) |
| **LSTM surrogate** | `LSTMSurrogate`, `SeqDS`, `train_lstm`, Huber on standardized ΔpH, gradient clipping |
| **Hybrid MDP** | `WastewaterMDP`: physics warm-start → LSTM, observation noise, curriculum masking, MC-Dropout penalty |
| **PPO** | `ppo_train` (custom PyTorch), not Stable-Baselines3 |
| **Baselines** | RBT, PID, LUT in notebook / `water_experiments_small`; **DDPG** small trainer for smoke comparisons |
| **Wilcoxon + Cohen’s d** | `wilcoxon_report`, Bonferroni helpers in experiments module |
| **Sec 2.3.2 diagnostic** | `validate_simulator_sec232` - one-step / short-window style check vs DS-5; **not** the full 50×120 min open-loop protocol |
| **Table 6 - automated gates (subset)** | Constants `TABLE6_GATE_*` in `water_methodology_impl`; `table6_gates()` compares RMSE, MAE(ΔpH), median Sec 2.3.2 error |
| **Notebook** | `methodology_implementation.ipynb` end-to-end: data → LSTM → gates → PPO → Tier-1 DCR plot → OOD shift plot → `methodology_first_pass_small` |
| **Figures** | `paper_figures.py` → `figures_paper/*.png` |

---

## Partially implemented (same idea, reduced scope)

| Manuscript | What we ship |
|------------|----------------|
| **Table 6 - six gates** | Code automates **three** thresholds (RMSE, MAE ΔpH, median Sec 2.3.2). **ECE**, **1000-step rollout drift**, etc. are **not** wired as automated pass/fail. |
| **Sec 2.3.2 full validation** | Word doc: **50×120 min** trajectories, MAPE acceptance. Repo: **short diagnostic** + **stabilized MAPE** fields (`MAPE_pct`, `MAPE_pct_median`); **median \|ΔpH\|** is the primary gate statistic. |
| **PPO budget** | Notebook uses **`DEMO_MODE = True`** by default (short LSTM epochs and PPO steps). Full **5×10⁶**-style runs require `DEMO_MODE = False` and time/GPU. |
| **Tier 1 - 500 episodes** | Demo uses **small** episode counts (e.g. 12 vs 80). |
| **Tier 2 / Tier 3** | **OOD shift figure** (DS-4 vs train scalers) is implemented. Full **Tier 2/3 episode protocols** from the Word doc are **not** fully replicated as separate 200-episode / DS-5 rollout studies. |
| **Dyna refresh** | Hook exists in `ppo_train` at a large step interval; not the focus of demo runs. |
| **Table 7 - GRU / TCN / MLP** | Mentioned in manuscript; **no** parallel training scripts for all architectures in-repo. |

---

## Data / labeling caveats (honest scope)

| Topic | Note |
|-------|------|
| **DS-3 “compliance”** | Labels are **heuristic** from proxies - not a regulatory adjudication. |
| **CUSUM on real DS-1** | Smooth NWIS windows may yield **zero** dosing windows → **all-null actions** and NaN inverse residual MAE in Table 3; methodology still valid, result is data-dependent. |
| **DS-5** | Bundled KU-MWQ **30 cm** (and optionally **60 cm**) workbook is used when present; cite **Mendeley DOI** in the paper. |

---

## Not implemented (future work / paper-only)

- **Stable-Baselines3 / Gymnasium** as in Table 18 - custom env + custom PPO instead.
- **MLflow / joblib pipeline serialization** as described in Sec 8 - not required for core reproducibility here.
- **ruptures** package CUSUM - internal CUSUM implementation.
- **Full ablation grid** (Table 17) with 3 seeds each.
- **Hyperparameter sensitivity sweeps** (Sec 13).
- **Statistical supplement** (bootstrap CIs, Holm-Bonferroni, etc.) beyond Wilcoxon helpers.

---

## How to run a “fuller” experiment

1. Set **`DEMO_MODE = False`** in the notebook (and allow long runtimes).
2. **Restart the Jupyter kernel** after any change to `water_*.py` so imports reload.
3. Keep **`WATER_TABLE2_REQUIRE_REAL=1`** if you want all Table-2 roles on public/bundled paths.
4. Record **git commit hash** and **`pip freeze`** (or `requirements-paper-lock.txt`) in supplementary material.

---

## File map (quick)

| File | Role |
|------|------|
| `water_methodology_impl.py` | Physics, preprocessing, LSTM, MDP, PPO |
| `water_rdi_loaders.py` | NWIS + CSV + KU-MWQ Table-2 assembly |
| `water_experiments_small.py` | Sec 2.3.2 diagnostic, Table 6 gates, smoke orchestration, DDPG helper |
| `paper_figures.py` | Publication plots |
| `methodology_implementation.ipynb` | End-to-end narrative |
| `Methodology_v01_Walkthrough.md` | Manuscript structure |
| `Results_Analysis_Figures.md` | Figure meanings |
