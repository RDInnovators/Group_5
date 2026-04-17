# Methodology v.01 — implementation status

This file is the **handoff map** for humans and agents: what is implemented in `Water/`, what is partial, and what remains to reach a “complete” alignment with *Methodology v.01* and a reproducible public-data story.

---

## Done (implemented in repo)

### Core pipeline (`water_methodology_impl.py`)

- **Stages 1–2:** Synthetic loaders remain available; **CUSUM** (Eq. 3) + **inverse titration** labelling (Eq. 4); **Table 3** reconstruction metrics; **preprocess_monitor** with Table 4 P1–P6 narrative (MinMax inputs, StandardScaler on ΔpH for LSTM, rolling min/max etc. where coded).
- **Physics / surrogate:** `f_titration`, **LSTM** training, **MC-Dropout** variance for Table 13-style gate, **σ** from DS-5 diffs + σ_model from val residuals.
- **RL:** `WastewaterMDP` (Table 8–10 style), **PPO** (`ppo_train`), curriculum masking, Dyna hook, hybrid physics→LSTM (Eq. 5), eval with `physics_warm=0` for surrogate rollouts.
- **Evaluation:** `rollout_policy`, **Wilcoxon** helpers; **RBT / PID** baselines in code paths used by the notebook / small experiments.
- **Table 2 naming:** `RDI_TABLE2`, `rdi_dataset_name`, `rdi_table2_lines()`, synth aliases aligned to methodology Table 2 labels.
- **`run_full_pipeline`:** Calls **`water_rdi_loaders.build_table2_mixed`** for DS-1…DS-5 (unless `WATER_USE_SYNTH_ONLY=1` or import/runtime failure → full synth fallback). Returns **`table2_flags`** dict indicating which datasets used public/real paths vs synth. Replaces DS-1 with synth if NWIS returns **fewer than 500** rows after load. Uses **real DS-5 IV** for σ when length and `pH` are sufficient, else synth Hz.

### Public / real data (`water_rdi_loaders.py`)

- **DS-1 / DS-5:** USGS NWIS **IV** (JSON), mapped to DS-1 schema (15‑min resample) and DS-5 wide stream with `pH` / `conductivity_uScm` names where codes exist.
- **DS-4:** NWIS **DV** → monthly medians. Tries **`WATER_NWIS_SITE_DS4`** if set, else **DS-1 site**, then **`WATER_NWIS_DV_FALLBACK_SITES`** (default includes **01646500** — Potomac at Point of Rocks, MD — which has daily **00400**+**00095** where Choptank **01491000** often lacks pH DV).
- **DS-2:** `WATER_DS2_CSV` optional; if unset, **auto-load** bundled `data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv` when present (~12k rows): WQP **USGS-MD** Ca + Mg → hardness proxy (2.497·Ca + 4.118·Mg mg/L as CaCO₃) merged with specific conductance on same **ActivityIdentifier** (from project `ds2_spc_MD_sample.csv` + downloaded Ca/Mg extracts used to build the paired file).
- **DS-3:** `WATER_DS3_CSV` optional; if unset, **auto-load** bundled `data/rdi/ds3_wqp_effluent_md_proxy.csv` (WQP **Effluent** pH, Maryland 2020–2021 partial export; `discharge_class` / `effluent_type` / `compliance_status` are **derived** from sample-fraction, org name, and pH/qualifier heuristics — not a regulatory adjudication).
- **Strict mode:** `WATER_TABLE2_REQUIRE_REAL=1` makes `build_table2_mixed` **raise** if any role is still synthetic (including `WATER_USE_SYNTH_ONLY=1`, which conflicts with strict mode). The notebook Stage 1 sets this by default.

### Notebook (`methodology_implementation.ipynb`)

- Stage 1 uses **`build_table2_mixed`**, prints **`TABLE2_FLAGS`**, keeps **`ds4`** and **`DS5_IV`** for downstream cells.
- **PPO σ cell:** prefers **real `DS5_IV`** when flags and length allow; else synth Hz.
- **Stage 6 OOD:** uses **`ds4` from Stage 1** (NWIS monthly when flag true) instead of always regenerating synth DS-4.
- **Stage 2a / evaluation:** Table 6 smoke, **LUT / PID / RBT**, Bonferroni α/3, boxplot `tick_labels`; imports **`water_experiments_small`** where needed.

### Short orchestration (`water_experiments_small.py`)

- **`methodology_first_pass_small`:** Chains **`run_full_pipeline`** (therefore mixed Table 2 when network + files OK) + Sec 2.3.2, Table 6 gates, tier smoke, DDPG, LUT, extended Wilcoxon / Cohen’s d, etc.

### Hygiene / fixes (from earlier threads)

- **`SeqDS` / `train_lstm`** unpacking and column passing fixes; **`estimate_AT_CT_from_ds2`** accepts scalar conductivity; **`WastewaterMDP`** constructor alignment; Jupyter kernel **`water-methodology`** registered (environment-specific).

---

## Partial / environment-dependent

| Item | Notes |
|------|--------|
| **DS-1 / DS-5 NWIS** | Real when `waterservices.usgs.gov` returns IV for `WATER_NWIS_SITE_DS1` (default `01491000`). Offline or API errors → synth. |
| **DS-4 NWIS DV** | Real when any tried site yields ≥12 monthly rows after merge (fallback chain avoids Choptank-only DV gaps). |
| **DS-2** | Real when bundled CSV exists (or `WATER_DS2_CSV` set). |
| **DS-3** | Real when bundled `ds3_wqp_effluent_md_proxy.csv` exists (or `WATER_DS3_CSV` set); columns beyond pH are **heuristic** labels on real effluent pH. |
| **Stage 2a long DS-5** | Still uses **`synth_ds5_hz`** for 100k–150k steps (length / stress test); not the same trace as short NWIS IV. |

---

## Next steps (to “complete” vs methodology + reproducibility)

Prioritize in this order unless the manuscript scope says otherwise.

1. **Align Stage 2a DS-5 with Table 2 narrative (optional)**  
   - Either: tile / repeat real NWIS IV to reach gate length, or: keep synth for gate and **document** explicitly in notebook + here that Sec 2.3.2 uses a long **synthetic** stress trace by design.

2. **Docstring / comment cleanup**  
   - Update `SYNTH_LOADERS` comment in `water_methodology_impl.py` (still implies “synthetic only”).  
   - Refresh `water_experiments_small.validate_simulator_sec232` docstring to mention mixed `run_full_pipeline` data.

3. **Automated regression**  
   - Test `build_table2_mixed` with `WATER_USE_SYNTH_ONLY=1` (deterministic flags).  
   - Optional smoke: `run_full_pipeline(demo_mode=True)` with timeout or mocked NWIS (harder).

4. **Manuscript-scale backlog (explicitly out of minimal code path)**  
   - Table 17 ablations, 3× seeds × 5M PPO steps, SB3/MLflow/Optuna — noted in module audit as optional tooling.

5. **Full OOD policy study (research, not one cell)**  
   - Notebook Stage 6 is a **preprocessor shift hook** on DS-4; full “train DS-1, evaluate policies on DS-4” is additional experiment design.

---

## Environment variables (quick reference)

| Variable | Effect |
|----------|--------|
| `WATER_USE_SYNTH_ONLY=1` | Skip NWIS and bundled CSV discovery — **all five** synthetic. Incompatible with `WATER_TABLE2_REQUIRE_REAL=1` (raises). |
| `WATER_DS2_CSV` | Override DS-2 path (columns: `hardness_mgL`, `conductivity_uScm`). |
| `WATER_DS3_CSV` | Real DS-3 cross-sectional table (four columns above). |
| `WATER_NWIS_SITE_DS1` | NWIS site for DS-1 + DS-5 IV (default `01491000`). |
| `WATER_NWIS_SITE_DS4` | Optional NWIS site for DS-4 DV first attempt (else tries DS-1 site, then fallbacks). |
| `WATER_NWIS_DV_FALLBACK_SITES` | Comma-separated USGS sites for DS-4 DV (default `01646500`). |
| `WATER_TABLE2_REQUIRE_REAL` | `1` / `true` / `yes` — fail if any Table-2 slot is synthetic. |

---

## Key paths

- Implementation: `water_methodology_impl.py`  
- Table 2 loaders: `water_rdi_loaders.py`  
- Bundled DS-2: `data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv`  
- Bundled DS-3: `data/rdi/ds3_wqp_effluent_md_proxy.csv`  
- Notebook: `methodology_implementation.ipynb`  
- Small smoke / first pass: `water_experiments_small.py`

---

*Last updated for agent handoff: aligns code + notebook behavior as of the “real Table 2 dataset” integration; adjust this file when you add DS-3, DS-4 defaults, or change loader contracts.*
