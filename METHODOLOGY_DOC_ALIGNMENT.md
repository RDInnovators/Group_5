# *Methodology v.01* (Word) ↔ repository alignment

This file maps the manuscript chapter **Methodology v.01.docx** to this codebase for **journal submission**. Use it when writing Methods / Supplementary so claims match what is implemented and what remains for the **Results** chapter.

---

## 1. Data inventory (Table 2)

| Manuscript | Implementation |
|------------|----------------|
| “RDI compendium” wording | **Public proxies:** USGS **NWIS** (IV/DV), EPA **WQP**-derived bundled CSVs (`data/rdi/`), **KU-MWQ** (Mendeley DOI [10.17632/34rczh25kc.4](https://data.mendeley.com/datasets/34rczh25kc.4)) for DS-5 when `WATER_DS5_SOURCE=auto` and the workbook is present. |
| DS-1–DS-5 roles | `water_rdi_loaders.build_table2_mixed`, `RDI_TABLE2`, `TABLE2_FLAGS`. |
| DS-3 “compliance” labels | Bundled proxy uses **heuristic** fields — **not** regulatory adjudication. State this in the paper. |

---

## 2. Physics simulator (Sec 2.3)

| Manuscript | Implementation |
|------------|----------------|
| Charge balance / `f_titration` | `water_methodology_impl.f_titration`, `V_SYSTEM_L`, `C_REAGENT_M`, action volumes `ACTION_VOLUMES_ML`. |
| A_T, C_T from DS-2 / DS-1 | `estimate_AT_CT_from_ds2`, used in `run_full_pipeline` / notebook. |
| Newton–Raphson \|ΔpH\| < 10⁻⁶ | Documented in module audit; solver in titration path. |

---

## 3. Sec 2.3.2 simulator validation vs DS-5

| Manuscript | Implementation |
|------------|----------------|
| **50** windows of **120 min**, open-loop (no dosing), MAPE / error **< 0.25 pH units** | **Not fully automated** as 50×120 min open-loop segments. `validate_simulator_sec232` is a **short diagnostic**: one-step comparison with **random** actions vs successive pH differences on the DS-5 series (reports median \|ΔpH\| and a MAPE-style statistic). **Primary Table 6 gate** for Sec 2.3.2 uses **median \|ΔpH\|** vs `TABLE6_GATE_SEC232_MEDIAN_DPH` (**0.25** pH units), matching the manuscript tolerance. |
| **Results section** | Report full 50×120 min protocol where reviewers expect it, or explicitly scope the submitted work to the **implemented** diagnostic and reserve the full study for future work / appendix. |

---

## 4. Table 6 — surrogate acceptance gates

Constants live in `water_methodology_impl.py` and are used by `water_experiments_small.table6_gates`:

| Gate (manuscript) | Constant | Value |
|-------------------|----------|-------|
| Surrogate vs simulator / val **RMSE** (pH) | `TABLE6_GATE_RMSE` | **0.10** |
| **MAE** on ΔpH residuals | `TABLE6_GATE_MAE_DPH` | **0.07** |
| Sec 2.3.2 median \|ΔpH\| (pH) | `TABLE6_GATE_SEC232_MEDIAN_DPH` | **0.25** |

Other Table 6 rows (physical clipping, 50 DS-5 trajectories MAPE, 1000-step rollout, ECE) require **additional metrics** not all wired to automated pass/fail in this repo — report in **Results** or extend `water_experiments_small` if you need full gate automation.

---

## 5. Preprocessing (Table 4)

Implemented in `preprocess_monitor`: P1–P6 narrative, chronological split, MinMax + StandardScaler on ΔpH for LSTM, rolling stats including **min/max** at 1h and 6h where coded. DS-4 held out of training in `run_full_pipeline`; DS-5 IoT may lack conductivity — **placeholders** are filled for preprocessing (`preprocess_monitor`).

---

## 6. LSTM surrogate (Table 5)

`train_lstm`: 2×LSTM 256→128, dropout 0.2, L=48, Huber δ=1, Adam + cosine — matches the table in the manuscript at the level implemented in code. Ablations (GRU, TCN, MLP in Table 7) are **not** all shipped as first-class trainers here.

---

## 7. Hybrid environment & PPO (Tables 8–13)

| Manuscript | Implementation |
|------------|----------------|
| Physics warm-start **100,000** steps | `run_full_pipeline(..., demo_mode=False)` uses `phys_warm = 100_000`; **demo_mode=True** uses shorter budgets for interactive runs. |
| PPO, curriculum, MC-Dropout gate | `ppo_train`, `WastewaterMDP`. |

---

## 8. Evaluation (Stage 5 / Table 16)

| Manuscript | Implementation |
|------------|----------------|
| **500-episode** test suite | Notebook uses **`n_ep = 12` when `DEMO_MODE` else `80`** (and similar smoke counts). For the paper, set **`DEMO_MODE = False`** and raise episode counts to match the manuscript **or** state explicitly that reported numbers use the **longer** configuration in supplementary material. |
| Wilcoxon + Bonferroni | `wilcoxon_report`, Bonferroni α/3 in notebook Stage 5; `methodology_first_pass_small` includes four comparisons. |

---

## 9. Six vs seven stages in the notebook

The Word doc **Table 1** lists **six** stages. The Jupyter notebook adds **Reproducibility & figure export** and **Stage 7** (`methodology_first_pass_small`) as **orchestration / smoke** — not a separate methodological stage in Table 1. Describe accordingly in the paper.

---

## 10. Publication checklist (methods paper)

1. Cite **data** (NWIS access date, sites; WQP; KU-MWQ DOI; bundled paths) — see `REPORT_AND_SUBMISSION.md`.
2. State **DS-3** heuristic labels and **Sec 2.3.2** scope (diagnostic vs full 50-window study).
3. Align **Table 6** reported numbers with `TABLE6_GATE_*` constants.
4. Align **compute** (`DEMO_MODE`, PPO steps) with what you actually ran for the camera-ready figures.
5. Tag a **git release** and archive **notebook HTML + `figures_paper/`** with the submission.

---

*Generated for handoff; update when the Word methodology or defaults change.*
