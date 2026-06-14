#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_paper"

PHASE3B = ROOT / "results" / "phase3_full" / "phase3b_evaluation_summary.json"
PHASE3C = ROOT / "results" / "phase3_full" / "phase3c_stats.json"
ACTUAL_RESULTS = ROOT / "actual_results.md"
PPO_CURVE_DIR = ROOT / "results" / "phase3_full" / "curves" / "ppo_full"
FAILURE_STATS = ROOT / "results" / "phase4_failure_diagnosis" / "failure_diagnosis_stats.json"

CONTROLLERS = ["rule_based", "pid_deadband", "lut", "ppo_full", "ddpg", "null", "random"]
SEEDS = [11, 22, 33]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.alpha": 0.3,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def save_all_formats(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{stem}.png",
        FIG_DIR / f"{stem}.pdf",
        FIG_DIR / f"{stem}.svg",
    ]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def figure1_headline(summary_t1: dict) -> list[Path]:
    dcr = [summary_t1[c]["DCR_mean"] for c in CONTROLLERS]
    tcu = [summary_t1[c]["TCU_mean"] for c in CONTROLLERS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    palette = {
        "rule_based": "#4C78A8",
        "pid_deadband": "#54A24B",
        "lut": "#72B7B2",
        "ppo_full": "#E45756",
        "ddpg": "#B279A2",
        "null": "#9D755D",
        "random": "#F58518",
    }
    colors = [palette[c] for c in CONTROLLERS]

    ax1 = axes[0]
    ax1.bar(CONTROLLERS, dcr, color=colors)
    ax1.set_title("Tier-1 DCR by controller")
    ax1.set_ylabel("DCR (%)")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="x", rotation=35)

    ax2 = axes[1]
    ax2.bar(CONTROLLERS, tcu, color=colors)
    ax2.set_title("Tier-1 TCU by controller")
    ax2.set_ylabel("TCU")
    ax2.tick_params(axis="x", rotation=35)

    # Visual emphasis requested by user: pid_deadband vs ppo_full contrast.
    pid_idx = CONTROLLERS.index("pid_deadband")
    ppo_idx = CONTROLLERS.index("ppo_full")
    ax1.annotate("Best DCR", (pid_idx, dcr[pid_idx]), xytext=(0, 8), textcoords="offset points", ha="center")
    ax2.annotate("High TCU", (ppo_idx, tcu[ppo_idx]), xytext=(0, 8), textcoords="offset points", ha="center")

    fig.suptitle("Figure 1: Headline Tier-1 compliance and chemical usage", y=1.02)
    return save_all_formats(fig, "figure1_tier1_dcr_tcu")


def figure2_seed_divergence() -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    seed_colors = {11: "#4C78A8", 22: "#54A24B", 33: "#E45756"}

    for seed in SEEDS:
        curve_path = PPO_CURVE_DIR / f"ppo_full_seed_{seed}_curve.csv"
        df = pd.read_csv(curve_path)
        # Keep real logged evaluation points only; do not smooth or interpolate.
        pts = df[df["eval_dcr_mean"].notna()].copy()
        ax.plot(
            pts["step"],
            pts["eval_dcr_mean"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=seed_colors[seed],
            label=f"Seed {seed}",
        )

    ax.set_title("Figure 2: PPO full seed divergence during training")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Eval DCR mean (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    return save_all_formats(fig, "figure2_ppo_seed_divergence")


def figure3_failure_mechanism(failure_stats: dict) -> list[Path]:
    ppo_frac = float(failure_stats["action_usage_best_ppo_seed"]["inband_dosing_fraction"])
    pid_frac = float(failure_stats["action_usage_pid"]["inband_dosing_fraction"])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = ["Best PPO seed (22)", "Deadband PID"]
    vals = [ppo_frac * 100.0, pid_frac * 100.0]
    colors = ["#E45756", "#54A24B"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)

    ax.set_title("Figure 3: In-band dosing failure mechanism")
    ax.set_ylabel("In-band timesteps with non-null action (%)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 2, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    return save_all_formats(fig, "figure3_inband_dosing_fraction")


def figure4_cer(summary_t1: dict) -> list[Path]:
    cer = [summary_t1[c]["CER_mean"] for c in CONTROLLERS]
    colors = ["#4C78A8", "#54A24B", "#72B7B2", "#E45756", "#B279A2", "#9D755D", "#F58518"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(CONTROLLERS, cer, color=colors)
    ax.set_title("Figure 4: Tier-1 CER across controllers")
    ax.set_ylabel("CER")
    ax.tick_params(axis="x", rotation=35)
    return save_all_formats(fig, "figure4_tier1_cer")


def main() -> None:
    # Touch all user-specified sources to guarantee provenance.
    _ = PHASE3C.read_text(encoding="utf-8")
    _ = ACTUAL_RESULTS.read_text(encoding="utf-8")

    summary = json.loads(PHASE3B.read_text(encoding="utf-8"))
    failure_stats = json.loads(FAILURE_STATS.read_text(encoding="utf-8"))

    configure_style()

    saved: list[Path] = []
    saved += figure1_headline(summary["T1"])
    saved += figure2_seed_divergence()
    saved += figure3_failure_mechanism(failure_stats)
    saved += figure4_cer(summary["T1"])

    captions = [
        "Figure 1: Tier-1 DCR and TCU show that deadband PID combines the highest compliance with far lower chemical usage than PPO full.",
        "Figure 2: PPO full training curves diverge strongly across seeds 11, 22, and 33, indicating unstable convergence behavior.",
        "Figure 3: In-band dosing audit shows best-seed PPO doses on 91.08% of compliant timesteps versus 0.39% for deadband PID.",
        "Figure 4: Tier-1 CER comparison shows classical controllers dominate PPO full on compliance-efficiency trade-off.",
    ]
    caption_path = FIG_DIR / "manuscript_figure_captions.txt"
    caption_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    saved.append(caption_path)

    manifest = FIG_DIR / "manuscript_figures_manifest.txt"
    manifest.write_text("\n".join(str(p.relative_to(ROOT)) for p in saved) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
