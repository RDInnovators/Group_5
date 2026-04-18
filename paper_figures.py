"""
Publication-style figures for `methodology_implementation.ipynb`.

Saves PNGs (300 dpi) under ``figures_paper/`` by default — suitable for LaTeX / Word.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
            "legend.fontsize": 9,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")


def extend_ds5_to_length(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Tile chronologically to reach ``n`` rows (for gate tests that need long streams)."""
    if len(df) >= n:
        return df.iloc[:n].copy()
    if len(df) < 1:
        raise ValueError("DS-5 frame is empty")
    reps = int(np.ceil(n / len(df)))
    out = pd.concat([df] * reps, ignore_index=True).iloc[:n].copy()
    return out


def plot_table2_overview(
    ds1: pd.DataFrame,
    ds5: pd.DataFrame,
    ds4: pd.DataFrame,
    table2_flags: Dict[str, bool],
    out_dir: Path,
    ds1_sample: int = 2500,
) -> None:
    """Multi-panel figures for DS-1, DS-5, DS-4 (Table 2 narrative)."""
    out_dir = Path(out_dir)
    # DS-1 — pH
    fig1, ax = plt.subplots(figsize=(7.2, 2.8))
    d1 = ds1.head(ds1_sample)
    ax.plot(d1["timestamp"], d1["pH"], lw=0.75, color="#1f77b4")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("pH")
    tag = "public" if table2_flags.get("DS-1") else "synthetic"
    ax.set_title(f"DS-1 — large-scale monitoring (15 min) [{tag}]")
    fig1.autofmt_xdate()
    savefig(fig1, out_dir / "fig_ds1_ph_timeseries.png")
    plt.close(fig1)

    # DS-5 — KU-MWQ or NWIS IV
    fig2, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    d5 = ds5.head(min(len(ds5), 6000))
    ts = d5["timestamp"]
    axes[0].plot(ts, d5["pH"], lw=0.55, color="#c0392b")
    axes[0].set_ylabel("pH")
    axes[1].plot(ts, d5["temperature_C"], lw=0.55, color="#e67e22")
    axes[1].set_ylabel("Temperature (°C)")
    if "turbidity_NTU" in d5.columns:
        axes[2].plot(ts, d5["turbidity_NTU"], lw=0.55, color="#27ae60")
        axes[2].set_ylabel("Turbidity (NTU)")
    elif "conductivity_uScm" in d5.columns:
        axes[2].plot(ts, d5["conductivity_uScm"], lw=0.55, color="#2980b9")
        axes[2].set_ylabel("Conductivity (µS/cm)")
    else:
        axes[2].set_ylabel("(no third channel)")
    tag5 = "public" if table2_flags.get("DS-5") else "synthetic"
    axes[0].set_title(f"DS-5 — high-frequency IoT proxy [{tag5}]")
    axes[2].set_xlabel("Time (UTC)")
    fig2.autofmt_xdate()
    savefig(fig2, out_dir / "fig_ds5_sensors.png")
    plt.close(fig2)

    # DS-4 — monthly pH
    fig3, ax = plt.subplots(figsize=(7.2, 2.8))
    if "timestamp" in ds4.columns and "pH" in ds4.columns:
        ax.plot(ds4["timestamp"], ds4["pH"], marker="o", ms=2.5, lw=0.9, color="#6c3483")
        ax.set_xlabel("Time (monthly)")
        ax.set_ylabel("pH")
        tag4 = "public" if table2_flags.get("DS-4") else "synthetic"
        ax.set_title(f"DS-4 — multi-decadal monthly proxy [{tag4}]")
        fig3.autofmt_xdate()
    savefig(fig3, out_dir / "fig_ds4_monthly_ph.png")
    plt.close(fig3)


def plot_dcr_boxplot(
    ppo_dcrs: np.ndarray,
    rbt_dcrs: np.ndarray,
    pid_dcrs: np.ndarray,
    lut_dcrs: np.ndarray,
    out_dir: Path,
    title: str = "Tier-1 discharge compliance rate (DCR)",
    *,
    close: bool = True,
) -> plt.Figure:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    data = [ppo_dcrs, rbt_dcrs, pid_dcrs, lut_dcrs]
    bp = ax.boxplot(data, tick_labels=["PPO", "RBT", "PID", "LUT"], patch_artist=True)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)
    ax.set_ylabel("DCR (%)")
    ax.set_title(title)
    savefig(fig, out_dir / "fig_dcr_boxplot.png")
    if close:
        plt.close(fig)
    return fig


def plot_ood_shift_bars(shift: Dict[str, float], out_dir: Path, max_bars: int = 10) -> None:
    """Bar chart of mean feature shift (DS-4 minus train) for OOD discussion."""
    out_dir = Path(out_dir)
    items = list(shift.items())[:max_bars]
    if not items:
        return
    labs, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(labs))
    ax.bar(x, vals, color="#5c4d7d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("Mean Δ (normalized space)")
    ax.set_title("OOD shift — DS-4 vs DS-1 training distribution (first features)")
    savefig(fig, out_dir / "fig_ood_feature_shift.png")
    plt.close(fig)


def plot_lstm_residual_hist(
    residuals: np.ndarray,
    out_dir: Path,
    title: str = "LSTM ΔpH residuals (validation)",
) -> None:
    """Histogram of validation residuals for supplementary material."""
    out_dir = Path(out_dir)
    x = np.asarray(residuals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.hist(x, bins=40, color="#3498db", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Predicted − true ΔpH")
    ax.set_ylabel("Count")
    ax.set_title(title)
    savefig(fig, out_dir / "fig_lstm_residual_hist.png")
    plt.close(fig)
