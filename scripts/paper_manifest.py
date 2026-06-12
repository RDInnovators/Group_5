#!/usr/bin/env python3
"""
Print versions and optional Table-2 loader probe for supplementary / reproducibility.
Run from repository root: python scripts/paper_manifest.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(git unavailable)"


def main() -> None:
    print("=== Water methodology - paper manifest ===\n")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"repo root: {ROOT}")
    print(f"git HEAD: {_git_head()}\n")

    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib", "openpyxl"):
        try:
            m = __import__(mod if mod != "sklearn" else "sklearn")
            ver = getattr(m, "__version__", "?")
            print(f" {mod}: {ver}")
        except ImportError as e:
            print(f" {mod}: NOT IMPORTABLE ({e})")

    # DISABLED (Phase 0 reconciliation): real-data table probing belongs to prior framing.
    # Keep this script focused on environment reproducibility only.
    print("\n--- Real-data table probe disabled in simulation-only Route A ---")

    print("\nDone.")


if __name__ == "__main__":
    main()
