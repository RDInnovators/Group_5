# Results analysis — figures in `figures_paper/`

This document explains **what each saved figure represents** and how to **interpret** it in a methods paper (Journal of Water Process Engineering–style framing). Numeric examples depend on your last notebook run; always cite **your** printed metrics next to each figure.

**Figure directory:** `figures_paper/` (300 dpi PNG from `paper_figures.py`, driven by `methodology_implementation.ipynb`).

---

## 1. `fig_ds1_ph_timeseries.png`

**What it is:** Time series of **pH** from **DS-1** (large-scale water-quality monitoring) at **15-minute** resolution, over the window loaded for the run.

**What it supports in the paper**

- Shows the **primary training signal** for the LSTM comes from **real public monitoring**, not synthetic data (if `TABLE2_FLAGS['DS-1']` is true).
- Motivates **why action reconstruction matters**: pH moves smoothly; **CUSUM** may or may not flag “dosing” events on a given river window — discuss if your Table 3 shows zero events.

**How to describe**

- “DS-1 pH trajectory used for surrogate training after preprocessing (Sec 2.5).”
- If compliance narrative: note the typical band vs your discharge target **[6.5, 8.5]** only if you overlay or reference summary stats (not always drawn on this plot).

---

## 2. `fig_ds5_sensors.png`

**What it is:** **Three stacked panels** — typically **pH**, **temperature**, and **turbidity** (or conductivity if turbidity missing) — from **DS-5** high-frequency IoT-style data (e.g. KU-MWQ **30 cm** when bundled).

**What it supports**

- **Q3 / sim-to-real** context: real sensor variability, diurnal patterns, and noise.
- **Noise model** motivation: σ from DS-5 diffs feeds the MDP (Table 13 narrative).

**How to describe**

- “Representative DS-5 high-frequency series used for noise characterization and Sec 2.3.2-style diagnostics.”
- Cite the **KU-MWQ dataset DOI** if that path was used.

---

## 3. `fig_ds4_monthly_ph.png`

**What it is:** **Monthly** pH from **DS-4** (global multi-decadal proxy, e.g. NWIS DV), long horizon.

**What it supports**

- **OOD** evaluation story: very different sampling and geography vs DS-1.
- Explains why **Tier 2** generalization is non-trivial (different distribution).

**How to describe**

- “DS-4 monthly pH — excluded from training; used for OOD diagnostics (Sec 7.1 Q2).”

---

## 4. `fig_dcr_boxplot.png`

**What it is:** **Box plots** of **DCR** (discharge compliance rate, %) for **PPO** vs **RBT, PID, LUT** over the **demo** episode count in the notebook.

**What it shows mechanically**

- DCR = fraction of episode steps with \(\mathrm{pH} \in [6.5, 8.5]\) (see `PH_LO` / `PH_HI` in code), expressed as percent.

**How to interpret**

- **Under demo budgets**, PPO may **underperform** classical baselines — that is a **valid result** (insufficient training, stochastic env, or policy not converged), not a plotting error.
- For publication claims, run **`DEMO_MODE = False`** and more episodes; report **mean ± SD** and **Wilcoxon** \(p\)-values printed in the notebook.

**Suggested paper wording**

- “Tier-1 surrogate-environment DCR distribution (pilot run); statistical tests reported for PPO vs each baseline with Bonferroni correction.”

---

## 5. `fig_ood_feature_shift.png`

**What it is:** **Bar chart** of **mean difference in normalized feature space**: **DS-4** minus **DS-1 training** mean, for the **first eight** overlapping monitor features (e.g. `pH_raw`, `dPH`, rolls, `cond`, `DO`).

**What it supports**

- **Q2** directly: quantifies **covariate shift** when OOD data are passed through **train-fitted** scalers (`fit=False`).
- Large shifts (often **conductivity**) support the need for **robust** or **conservative** policies under distribution shift.

**How to describe**

- “OOD mean feature shift (DS-4 vs DS-1 train) after MinMax scaling fit on DS-1 train only.”
- Avoid over-claiming: this is **first-moment shift**, not a full distributional test.

---

## Linking figures to evaluation questions (manuscript Sec 7.1)

| Question | Primary evidence in this repo |
|----------|-------------------------------|
| **Q1** — vs baselines | `fig_dcr_boxplot.png` + printed Wilcoxon lines |
| **Q2** — OOD | `fig_ood_feature_shift.png` + optional narrative on DS-4 init in extended runs |
| **Q3** — sim-to-real | DS-5 plots + Sec 2.3.2 metrics + DS-5 LSTM gate metrics in notebook output |

---

## Metrics to report next to figures (from the notebook)

Copy from your run (values change with seed and `DEMO_MODE`):

- **Table 6–style gates:** surrogate val RMSE / MAE(ΔpH), Sec 2.3.2 **median \|ΔpH\|**, pass/fail vs `TABLE6_GATE_*`.
- **Sec 2.3.2 dict:** `MAPE_pct`, `MAPE_pct_median`, `median_abs_dph_err` — emphasize **median absolute error** as the stable gate; MAPE is auxiliary.
- **MC-Dropout:** uncertainty **p95** used as RL penalty threshold.
- **Tier-1 smoke:** `methodology_first_pass_small` prints consolidated keys (`tier1`, `table6_gates`, etc.).

---

## Reproducibility checklist for the “Results” section

1. State **`DEMO_MODE`** and approximate **compute** (CPU/GPU, runtime).
2. State **data paths** (NWIS sites, KU-MWQ file present or NWIS fallback).
3. **Kernel restart** after code edits (otherwise metrics can reflect stale imports).
4. Commit **git hash** and environment file alongside supplementary material.

---

*For the full methodological intent of each stage, see `Methodology_v01_Walkthrough.md`. For code vs manuscript gaps, see `Implementation_and_Remaining_Work.md`.*
