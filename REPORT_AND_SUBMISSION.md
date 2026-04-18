# Methods paper — writing handoff & journal checklist

This document is the **single entry point** for co-authors writing the report: what the code does, which data to cite, how to describe limitations honestly, and how to claim reproducibility. Keep it aligned with the repository as you revise the manuscript.

**Reduced-scope freeze (what is / is not in scope for drafting):** [`WRITER_HANDOFF.md`](WRITER_HANDOFF.md).

**Manuscript ↔ code:** [`METHODOLOGY_DOC_ALIGNMENT.md`](METHODOLOGY_DOC_ALIGNMENT.md) maps *Methodology v.01.docx* to this repo (Table 6 constants, Sec 2.3.2 scope, `DEMO_MODE` vs full budgets).

---

## 1. Contribution framing (methods paper)

**Safe core claims**

- A **reproducible pipeline** for MBRL wastewater dosing research: public Table-2-style data assembly (loaders), CUSUM / inverse titration labelling, LSTM surrogate, hybrid physics–LSTM MDP, PPO training, and evaluation hooks documented in `methodology_implementation.ipynb`.
- **Transparency** on data sources (NWIS, WQP-derived CSVs, KU-MWQ) and on **heuristic** fields where applicable (DS-3).

**Avoid over-claiming**

- Do not imply **regulatory compliance** labels from DS-3 are ground truth (see §4).
- Do not imply **all** notebook stress paths use the same long **real** DS-5 trace unless you verify each cell (see `METHODOLOGY_STATUS.md`, Stage 2a long DS-5 note).

---

## 2. Canonical configuration for the “paper run”

Use this so the manuscript matches one reproducible configuration.

| Setting | Value | Note |
|--------|--------|------|
| `WATER_TABLE2_REQUIRE_REAL` | `1` | Fail if any Table-2 role is synthetic (Stage 1 in notebook sets this by default). |
| `WATER_DS5_SOURCE` | `auto` or `ku_mwq` | **auto**: use bundled KU-MWQ 30 cm Excel when `data/rdi/.../Sensor data for 30 cm.xlsx` exists; else NWIS IV. |
| `WATER_USE_SYNTH_ONLY` | unset / `0` | Must not be `1` when strict real mode is on. |
| Network | Online | NWIS IV/DV must be reachable for DS-1 / DS-4 (and for DS-5 if KU file absent and you rely on NWIS). |

**Bundled files authors should name in Data availability**

- DS-2: `data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv`
- DS-3: `data/rdi/ds3_wqp_effluent_md_proxy.csv`
- DS-5 (KU-MWQ): `data/rdi/KU-MWQ A Dataset for Monitoring Water Quality Using Digital Sensors/Sensor data for 30 cm.xlsx`

---

## 3. Table 2 — copy-paste summary for the manuscript

| ID | Role in pipeline | Implementation source | Cite / pointer |
|----|-------------------|-------------------------|----------------|
| DS-1 | Large-scale monitoring–style surface water | USGS NWIS **instantaneous** (IV), resampled to 15 min in code | [USGS Water Services](https://waterservices.usgs.gov/); site from `WATER_NWIS_SITE_DS1` (default `01491000`). |
| DS-2 | Regional hardness–conductivity pairing | Bundled WQP-derived CSV (USGS-MD Ca+Mg → hardness; SC on same activity) | Describe as WQP-derived; file path above. |
| DS-3 | Effluent / survey-style cross-section | Bundled WQP effluent pH proxy CSV; **derived** categorical fields | Same; **limitations** in §4. |
| DS-4 | Multi-decadal monthly proxy | USGS NWIS **daily** (DV) aggregated to monthly; fallback site chain in code | NWIS DV; sites from env / defaults in `water_rdi_loaders.py`. |
| DS-5 | High-frequency IoT-style sensors | **KU-MWQ** 30 cm workbook (pH, temperature, turbidity) when selected; else NWIS IV | **Nahid et al., Mendeley Data**, DOI **10.17632/34rczh25kc.4** ([dataset](https://data.mendeley.com/datasets/34rczh25kc/4)), **CC BY 4.0**. |

---

## 4. Limitations (text you can adapt)

**DS-3.** Fields such as `discharge_class`, `effluent_type`, and `compliance_status` in the bundled proxy are **heuristic** (derived from sample metadata and pH-related rules), **not** regulatory adjudications. Use language such as “proxy labels for algorithm development” unless you replace them with verified records.

**DS-5.** KU-MWQ is **~1 sample per minute** (pond, fixed deployment), not USGS stream network data. It matches the **IoT / high-frequency** narrative but is **not** interchangeable with NWIS without rewriting methods. If the manuscript compares NWIS vs KU-MWQ, run `WATER_DS5_SOURCE=nwis` vs `ku_mwq` explicitly.

**NWIS.** Access is **live**; retrieved series depend on site availability and USGS service responses. Record **access date** and **site IDs** in supplementary material.

**Computational variance.** PPO and other stochastic components may differ across runs unless seeds are fixed everywhere (see code and notebook); for strict claims, report **seeds** and ideally **multiple runs** if the venue expects uncertainty quantification.

---

## 5. Citations (starter set)

1. **KU-MWQ (DS-5).** Nahid, A.-A., Arafat, A. I., Akter, T., Ahammed, M. F., Ali, M. Y., & Ali, M. Y. (2020). *KU-MWQ: A Dataset for Monitoring Water Quality Using Digital Sensors* (Version 4) [Data set]. Mendeley Data. https://doi.org/10.17632/34rczh25kc.4 (CC BY 4.0).

2. **USGS NWIS Water Services** (DS-1 IV, DS-4 DV, optional DS-5 IV). U.S. Geological Survey. *National Water Information System web services.* https://waterservices.usgs.gov/ (access date: ______).

3. **WQP** (conceptual provenance for DS-2 / DS-3 CSV construction). U.S. EPA. *Water Quality Portal.* https://www.waterqualitydata.us/ (access date: ______).

Authors should fill **access dates** and any **journal-specific** citation style.

---

## 6. Draft — Data availability statement

*Adapt for your journal’s word limit and policy.*

> The implementation code and bundled tabular extracts used for DS-2 and DS-3 are available in the project repository under `data/rdi/` with filenames listed in `README.md` and this file. DS-1, DS-4, and (when DS-5 is not taken from the bundled Excel) DS-5 series are retrieved at run time from **U.S. Geological Survey National Water Information System** web services (`waterservices.usgs.gov`); exact site identifiers and parameters follow environment variables documented in `README.md`. The high-frequency IoT-style DS-5 stream used when `WATER_DS5_SOURCE` selects KU-MWQ is from **Nahid et al., Mendeley Data**, DOI 10.17632/34rczh25kc.4 (CC BY 4.0). Analysis scripts and the full notebook workflow are provided as `methodology_implementation.ipynb` together with Python modules `water_methodology_impl.py`, `water_rdi_loaders.py`, and `water_experiments_small.py`.

---

## 7. Reproducibility — what to archive for reviewers

1. **Environment:** Python version + dependencies. Install from `requirements_methodology.txt`; for stricter replication, use `requirements-paper-lock.txt` (pinned core stack from a reference venv).
2. **Code state:** Git tag or commit hash (e.g. `git rev-parse HEAD`) recorded in supplementary material.
3. **Manifest:** Run `python scripts/paper_manifest.py` from the `Water/` directory and paste or attach the output (versions + optional Table-2 probe).
4. **Notebook:** Run `methodology_implementation.ipynb` top to bottom after **Publication setup**; export HTML/PDF for appendix or Zenodo bundle.

### Notebook-generated figures (`figures_paper/`)

| File | Content |
|------|---------|
| `fig_ds1_ph_timeseries.png` | DS-1 pH vs time (NWIS IV, 15 min) |
| `fig_ds5_sensors.png` | DS-5 pH, temperature, turbidity (or conductivity if NWIS) |
| `fig_ds4_monthly_ph.png` | DS-4 monthly pH (NWIS DV proxy) |
| `fig_dcr_boxplot.png` | Tier-1 DCR: PPO vs RBT / PID / LUT |
| `fig_ood_feature_shift.png` | Bar chart — mean feature shift DS-4 vs train |

Caption each figure with **data source** (site IDs, KU-MWQ DOI, access date for NWIS).

---

## 8. Suggested report outline (methods-heavy)

1. **Introduction** — problem: dosing under uncertainty; gap: need for reproducible MBRL + public-data story.
2. **Data** — Table 2 roles; NWIS / WQP / KU-MWQ; limitations (§4).
3. **Methods** — labelling (CUSUM + inverse titration), preprocessing (Table 4 narrative), surrogate (LSTM, MC-Dropout), MDP, PPO, baselines (PID/RBT/LUT as implemented).
4. **Experimental protocol** — notebook stages; metrics (Table 3/6-style); what is smoke vs full budget.
5. **Results** — figures/tables from notebook; no claims beyond what cells compute.
6. **Discussion** — sim-to-real, DS-3 proxy nature, NWIS vs pond IoT generalization.
7. **Conclusion & future work** — extended experiments (seeds, full PPO budget) per `METHODOLOGY_STATUS.md` / `METHODOLOGY_DOC_ALIGNMENT.md`.

---

## 9. Co-author task list

| Task | Owner | Notes |
|------|--------|------|
| Fill USGS/WQP **access dates** and **site list** used in the run | Data / methods | From notebook output + env |
| Finalize **Data availability** + **code availability** | Corresponding author | Repo URL + tag |
| **Limitations** paragraph | All | DS-3 + optional synthetic stress paths |
| **Figure captions** with data source per panel | Writing | DS-1…DS-5 |
| **Supplementary** HTML notebook | Tech | After final figures |

---

## 10. Quick commands

```bash
cd Water
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements_methodology.txt
python scripts/paper_manifest.py
jupyter notebook methodology_implementation.ipynb
```

---

*Maintained for the Group_5 / Water methodology track; update this file when loaders, defaults, or datasets change.*
