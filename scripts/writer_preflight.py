#!/usr/bin/env python3
"""
Sanity checks before handing the repo to writers (no NWIS required for basic pass).

Usage:  python scripts/writer_preflight.py
Run from the Water/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    errors: list[str] = []
    print("=== Writer preflight (reduced-scope freeze) ===\n")

    # Core imports (must work after `pip install -r requirements_methodology.txt`)
    try:
        import water_methodology_impl as m
        import water_rdi_loaders as L
        import water_experiments_small as ex
    except Exception as e:
        errors.append(f"core import failed: {e}")
        print("FAIL:", e)
        return 1

    try:
        import paper_figures  # noqa: F401 — pulls matplotlib; notebook figure export needs this
    except ImportError as e:
        print(f"  WARNING: paper_figures not importable (install deps): {e}")

    for name, val in (
        ("TABLE6_GATE_RMSE", getattr(m, "TABLE6_GATE_RMSE", None)),
        ("TABLE6_GATE_MAE_DPH", getattr(m, "TABLE6_GATE_MAE_DPH", None)),
        ("TABLE6_GATE_SEC232_MEDIAN_DPH", getattr(m, "TABLE6_GATE_SEC232_MEDIAN_DPH", None)),
    ):
        if val is None:
            errors.append(f"missing {name}")
        else:
            print(f"  {name} = {val}")

    ku = L.ku_mwq_default_xlsx()
    print(f"  KU-MWQ default path exists: {ku.is_file()} ({ku})")

    for p in (
        ROOT / "README.md",
        ROOT / "Methodology_v01_Walkthrough.md",
        ROOT / "Implementation_and_Remaining_Work.md",
        ROOT / "Results_Analysis_Figures.md",
        ROOT / "methodology_implementation.ipynb",
    ):
        ok = p.is_file()
        print(f"  {'OK' if ok else 'MISSING'} {p.relative_to(ROOT)}")
        if not ok:
            errors.append(f"missing {p}")

    g = ex.table6_gates(0.05, 0.05, 0.1)
    if "thresholds" not in g:
        errors.append("table6_gates broken")
    else:
        print(f"  table6_gates sample thresholds: {g['thresholds']}")

    print()
    if errors:
        print("FAILED:", errors)
        return 1
    print("PASS — ready for writing handoff (run full notebook separately for figures/numbers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
