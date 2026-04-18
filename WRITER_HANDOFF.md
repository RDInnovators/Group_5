# Writer handoff — **reduced scope** (frozen for drafting)

**Status:** Implementation is **complete for the agreed reduced scope**. Co-authors can start the manuscript from this file; use the linked docs for detail and citations.

**Reduced scope means:** we ship a **reproducible end-to-end pipeline** (public data → labelling → physics + LSTM → MDP/PPO → baselines → figures) without implementing every optional experiment in *Methodology v.01.docx* (full Sec 2.3.2 window study, all Table 6/ECE/rollout gates, Table 7 architecture ablations, 500-episode runs at full budget). Those gaps are listed in [`METHODOLOGY_DOC_ALIGNMENT.md`](METHODOLOGY_DOC_ALIGNMENT.md) and must be framed as **limitations** or **future work**, not as completed results.

---

## What is done (safe to describe as implemented)

| Piece | Location |
|-------|----------|
| Table 2 mixed loaders (NWIS, WQP CSVs, KU-MWQ DS-5 option) | `water_rdi_loaders.py`, env vars in `README.md` |
| CUSUM + inverse titration, Table 3 metrics | `water_methodology_impl.py` |
| Preprocessing P1–P6, chronological split | `preprocess_monitor`, notebook Stage 1d |
| Physics titration `f_titration`, hybrid MDP, PPO | `water_methodology_impl.py`, notebook Stages 3–5 |
| LSTM surrogate (Table 5–style) | `train_lstm` |
| Table 6 **numeric gates** aligned to manuscript | `TABLE6_GATE_RMSE`, `TABLE6_GATE_MAE_DPH`, `TABLE6_GATE_SEC232_MEDIAN_DPH` in `water_methodology_impl.py`; `table6_gates()` in `water_experiments_small.py` |
| Sec 2.3.2 **diagnostic** (not full 50×120 min protocol) | `validate_simulator_sec232` — see alignment doc |
| Baselines (RBT, PID, LUT), Wilcoxon, smoke orchestration | Notebook Stage 5; `methodology_first_pass_small` in Stage 7 |
| Publication figures (300 dpi PNG) | `figures_paper/` after running notebook (`paper_figures.py`) |
| Data & citation draft text | [`REPORT_AND_SUBMISSION.md`](REPORT_AND_SUBMISSION.md) |
| Word doc ↔ code mapping | [`METHODOLOGY_DOC_ALIGNMENT.md`](METHODOLOGY_DOC_ALIGNMENT.md) |

---

## What writers must **not** claim as fully implemented

- **DS-3 “compliance”** — heuristic proxy labels, not regulatory ground truth.
- **Full Sec 2.3.2** per Word (50 windows × 120 min open-loop) — only the **short diagnostic** exists in code.
- **Every Table 6 row** (ECE, 1000-step rollout cap, etc.) — only the **three automated gates** above are wired; say “additional criteria left to extended study” or report manually if you add experiments later.
- **Table 7** (GRU/TCN/MLP) — not provided as parallel trainers in-repo.
- **“500 episodes” / full PPO budget** — notebook defaults use **`DEMO_MODE = True`** unless you change it; **state clearly** what budget your numbers used.

---

## One reproduction path (before citing numbers)

```bash
cd Water
python3 -m venv .venv && source .venv/bin/activate   # if needed
pip install -r requirements_methodology.txt
python scripts/writer_preflight.py    # sanity check (no NWIS required for basic pass)
jupyter notebook methodology_implementation.ipynb
```

Run **all cells top to bottom** (after **Reproducibility & figure export**). Needs **network** for NWIS if using real Table-2 data. Then:

```bash
python scripts/paper_manifest.py      # versions + optional Table-2 probe (needs network)
```

Record **git commit hash** and **Python + key package versions** in supplementary material.

---

## Files co-authors should read (order)

1. **This file** — scope boundary  
2. [`REPORT_AND_SUBMISSION.md`](REPORT_AND_SUBMISSION.md) — citations, data availability draft, limitations language  
3. [`METHODOLOGY_DOC_ALIGNMENT.md`](METHODOLOGY_DOC_ALIGNMENT.md) — manuscript vs code  
4. [`METHODOLOGY_STATUS.md`](METHODOLOGY_STATUS.md) — technical detail / env table  
5. [`README.md`](README.md) — layout and env vars  

---

## Journal note (JWPE)

Frame the paper as **wastewater process control** (adaptive dosing, compliance) with **open, citable data** and a **reproducible workflow**—not as exhaustive benchmark of every RL variant. See prior discussion on **Journal of Water Process Engineering** focus.

---

*Last updated: reduced-scope freeze for writing. Implementation changes after this should update `METHODOLOGY_DOC_ALIGNMENT.md` and this section.*
