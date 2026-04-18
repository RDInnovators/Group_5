# Water - MBRL wastewater dosing (Methodology v.01)

Python implementation and Jupyter walkthrough for the methodology pipeline.

## Documentation

| Document | Contents |
|----------|----------|
| [`Methodology_v01_Walkthrough.md`](Methodology_v01_Walkthrough.md) | Step-by-step summary of *Methodology v.01* (datasets, physics, CUSUM, preprocessing, LSTM, MDP, PPO, evaluation). |
| [`Implementation_and_Remaining_Work.md`](Implementation_and_Remaining_Work.md) | What this repository implements vs the Word manuscript, and what remains future work. |
| [`Results_Analysis_Figures.md`](Results_Analysis_Figures.md) | What each figure in `figures_paper/` means and how to interpret it in a paper. |

Keep your local copy of **`Methodology v.01.docx`** beside the repo if you need the full tables and equations; it is not committed to Git.

## Quick start

```bash
cd Water
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_methodology.txt
jupyter notebook methodology_implementation.ipynb
```

See `requirements_methodology.txt` and `.env.example` for data-source environment variables.
