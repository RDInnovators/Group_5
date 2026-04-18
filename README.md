# Water — MBRL wastewater dosing (*Methodology v.01*)

Python implementation and Jupyter walkthrough for the methodology pipeline: Table 2 data loaders (USGS NWIS + WQP-backed CSVs), CUSUM / inverse titration labelling, LSTM surrogate, hybrid physics–LSTM MDP, and PPO training.

## For report authors & journal submission

**Start here (reduced-scope handoff):** [`WRITER_HANDOFF.md`](WRITER_HANDOFF.md) — what is frozen for drafting, what not to claim, reproduction path, and links to the rest.

Then: [`REPORT_AND_SUBMISSION.md`](REPORT_AND_SUBMISSION.md) — Table 2 citations, limitations text, draft data-availability statement, co-author checklist, and canonical env settings for the methods paper.

- **Pinned deps (optional):** `requirements-paper-lock.txt` (core stack). Day-to-day: `requirements_methodology.txt`.
- **One-page manifest (versions + flags):** `python scripts/paper_manifest.py` from the `Water/` directory.

## Quick start

```bash
cd Water
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_methodology.txt
```

Open `methodology_implementation.ipynb` in Jupyter or VS Code. Select the interpreter from `.venv`.

## Public data vs synthetic

By default the notebook sets `WATER_TABLE2_REQUIRE_REAL=1` so **all five Table-2 roles** must use public data (NWIS + bundled CSVs under `data/rdi/`, and **DS-5** from the bundled [KU-MWQ](https://data.mendeley.com/datasets/34rczh25kc/4) Excel when `WATER_DS5_SOURCE=auto` and the file is present). For offline work, comment out that line in Stage 1.

Details, env vars, and known gaps: see [`METHODOLOGY_STATUS.md`](METHODOLOGY_STATUS.md).

## Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `WATER_USE_SYNTH_ONLY=1` | Force synthetic data only (conflicts with `WATER_TABLE2_REQUIRE_REAL`). |
| `WATER_TABLE2_REQUIRE_REAL=1` | Fail if any dataset falls back to synthetic. |
| `WATER_DS2_CSV`, `WATER_DS3_CSV` | Override bundled DS-2 / DS-3 CSV paths. |
| `WATER_NWIS_SITE_DS1`, `WATER_NWIS_SITE_DS4` | USGS site numbers for IV / DV. |
| `WATER_NWIS_DV_FALLBACK_SITES` | Comma-separated DV sites for DS-4 (default includes `01646500`). |
| `WATER_DS5_SOURCE` | `auto` (default): use bundled KU-MWQ 30 cm `.xlsx` for DS-5 if present, else NWIS IV. `ku_mwq` / `nwis` force one source. |
| `WATER_DS5_KU_MWQ_XLSX` | Optional path to the KU-MWQ 30 cm workbook (pH + sensors). |

## Repository layout

| Path | Description |
|------|-------------|
| `water_methodology_impl.py` | Core pipeline, MDP, PPO, LSTM, preprocessing. |
| `water_rdi_loaders.py` | NWIS + CSV Table-2 builder (`build_table2_mixed`). |
| `water_experiments_small.py` | Short “first pass” smoke + baselines. |
| `data/rdi/` | Bundled real-proxy CSVs (WQP-derived DS-2 / DS-3). |
| `methodology_implementation.ipynb` | End-to-end notebook. |
| `WRITER_HANDOFF.md` | **Reduced-scope freeze** for co-authors: what is done, what not to claim. |
| `REPORT_AND_SUBMISSION.md` | **Writing handoff:** citations, limitations, data availability draft, checklist. |
| `METHODOLOGY_DOC_ALIGNMENT.md` | *Methodology v.01.docx* ↔ code (Table 6 gates, Sec 2.3.2, scopes). |
| `scripts/writer_preflight.py` | Quick import / path sanity check before citing numbers. |
| `scripts/paper_manifest.py` | Print Python/package versions + optional Table-2 loader probe. |
| `paper_figures.py` | Publication-style plots used by the notebook (`figures_paper/*.png`). |
| `requirements-paper-lock.txt` | Optional pinned core stack for replication. |

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

5. **`Methodology v.01.docx`** is **not** included in this repository (keep your copy locally; see `METHODOLOGY_DOC_ALIGNMENT.md` for manuscript ↔ code mapping).

6. Optional: add a **LICENSE** file and clarify terms for WQP/USGS-derived CSVs and third-party datasets (e.g. KU-MWQ, CC BY 4.0).

## Data provenance

- **USGS NWIS:** [waterservices.usgs.gov](https://waterservices.usgs.gov/) (User-Agent set in code).
- **Bundled CSVs:** Built from EPA WQP-style exports and NWIS-accessible fields; see `METHODOLOGY_STATUS.md` for DS-2 / DS-3 construction notes.
- **KU-MWQ (DS-5 option):** [Mendeley Data 10.17632/34rczh25kc.4](https://data.mendeley.com/datasets/34rczh25kc/4) (CC BY 4.0); 30 cm sheet is read via `openpyxl`.

## Requirements

See `requirements_methodology.txt`. GPU is optional; CPU runs are supported for `demo_mode=True`.
