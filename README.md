# Water — MBRL wastewater dosing (*Methodology v.01*)

Python implementation and Jupyter walkthrough for the methodology pipeline: Table 2 data loaders (USGS NWIS + WQP-backed CSVs), CUSUM / inverse titration labelling, LSTM surrogate, hybrid physics–LSTM MDP, and PPO training.

## Quick start

```bash
cd Water
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_methodology.txt
```

Open `methodology_implementation.ipynb` in Jupyter or VS Code. Select the interpreter from `.venv`.

## Public data vs synthetic

By default the notebook sets `WATER_TABLE2_REQUIRE_REAL=1` so **all five Table-2 roles** must use public data (NWIS + bundled CSVs under `data/rdi/`). For offline work, comment out that line in Stage 1.

Details, env vars, and known gaps: see [`METHODOLOGY_STATUS.md`](METHODOLOGY_STATUS.md).

## Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `WATER_USE_SYNTH_ONLY=1` | Force synthetic data only (conflicts with `WATER_TABLE2_REQUIRE_REAL`). |
| `WATER_TABLE2_REQUIRE_REAL=1` | Fail if any dataset falls back to synthetic. |
| `WATER_DS2_CSV`, `WATER_DS3_CSV` | Override bundled DS-2 / DS-3 CSV paths. |
| `WATER_NWIS_SITE_DS1`, `WATER_NWIS_SITE_DS4` | USGS site numbers for IV / DV. |
| `WATER_NWIS_DV_FALLBACK_SITES` | Comma-separated DV sites for DS-4 (default includes `01646500`). |

## Repository layout

| Path | Description |
|------|-------------|
| `water_methodology_impl.py` | Core pipeline, MDP, PPO, LSTM, preprocessing. |
| `water_rdi_loaders.py` | NWIS + CSV Table-2 builder (`build_table2_mixed`). |
| `water_experiments_small.py` | Short “first pass” smoke + baselines. |
| `data/rdi/` | Bundled real-proxy CSVs (WQP-derived DS-2 / DS-3). |
| `methodology_implementation.ipynb` | End-to-end notebook. |
| `Methodology v.01.docx` | Source methodology document (verify you have rights before publishing). |

## Uploading to GitHub

This repo is **already initialized** with `main` and an initial commit (if you cloned before that step, run `git init` yourself).

1. Set your **commit identity** inside this repo (or use `--global`) if Git rejected the first commit:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
# Optional: fix the placeholder author on the first commit
git commit --amend --reset-author --no-edit
```

2. Create an empty repository on GitHub (no README/license if you will push this tree).

3. From this directory:

```bash
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

4. Confirm `.venv` is **not** tracked (`git ls-files` must not list `.venv/`). The `.gitignore` excludes it.

5. **`Methodology v.01.docx`** is currently **untracked**. Add it only if you have redistribution rights:  
   `git add "Methodology v.01.docx" && git commit -m "Add methodology document"`

6. Optional: add a **LICENSE** file and clarify terms for the Word doc and WQP/USGS-derived CSVs.

## Data provenance

- **USGS NWIS:** [waterservices.usgs.gov](https://waterservices.usgs.gov/) (User-Agent set in code).
- **Bundled CSVs:** Built from EPA WQP-style exports and NWIS-accessible fields; see `METHODOLOGY_STATUS.md` for DS-2 / DS-3 construction notes.

## Requirements

See `requirements_methodology.txt`. GPU is optional; CPU runs are supported for `demo_mode=True`.
