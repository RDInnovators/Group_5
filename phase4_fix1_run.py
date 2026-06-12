#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import phase2_baselines_mechanism as p2
import water_experiments_small as wes
import water_methodology_impl as m


SEEDS = (11, 22, 33)
OUT_ROOT = Path("results/phase4_fix1")
MODEL_DIR = OUT_ROOT / "models" / "ppo_full"
CURVE_DIR = OUT_ROOT / "curves" / "ppo_full"
EVAL_DIR = OUT_ROOT / "eval"


def _load_ppo(path: Path, device: torch.device) -> m.ActorCritic:
    ckpt = torch.load(path, map_location=device)
    model = m.ActorCritic().to(device)
    model.load_state_dict(ckpt["state_dict"])
    if "obs_rms_mean" in ckpt:
        model.obs_rms.mean = ckpt["obs_rms_mean"].to(device)
        model.obs_rms.var = ckpt["obs_rms_var"].to(device)
        model.obs_rms.count = float(ckpt["obs_rms_count"])
    model.eval()
    return model


def _rollout_episode(
    model: m.ActorCritic,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    env = p2.make_sim_env(enable_curriculum_masking=False, enable_escalation_penalty=True, A_T=1.5, C_T=1.0)
    obs = env.reset(rng).astype(np.float32)
    phs = [env.ph]
    acts = []
    inband = 0
    inband_dose = 0
    for _ in range(m.T_MAX):
        ph = env.ph
        if m.PH_LO <= ph <= m.PH_HI:
            inband += 1
        with torch.no_grad():
            o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            dist, _ = model(o, torch.ones(1, 11, device=device))
            a = int(dist.sample().item())
        if (m.PH_LO <= ph <= m.PH_HI) and (a != 0):
            inband_dose += 1
        obs, _, done, _ = env.step(a, rng)
        obs = obs.astype(np.float32)
        acts.append(a)
        phs.append(env.ph)
        if done:
            break
    ph = np.asarray(phs, dtype=np.float64)
    doses = np.asarray([m.ACTION_VOLUMES_ML[int(a)] for a in acts], dtype=np.float64)
    dcr = float(np.mean((ph >= m.PH_LO) & (ph <= m.PH_HI)) * 100.0)
    tcu = float(np.sum(doses))
    cer = float(dcr / (1.0 + tcu))
    return {
        "DCR": dcr,
        "TCU": tcu,
        "CER": cer,
        "inband_steps": float(inband),
        "inband_dose_steps": float(inband_dose),
        "inband_dosing_fraction": float(inband_dose / max(1, inband)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).resolve().parent / "train_one_job_fix1.py"
    # Train/reuse 3 seeds.
    for seed in SEEDS:
        model_path = MODEL_DIR / f"ppo_full_seed_{seed}.pt"
        curve_path = CURVE_DIR / f"ppo_full_seed_{seed}_curve.csv"
        if model_path.exists() and curve_path.exists():
            print(f"[phase4_fix1] seed={seed} reuse existing artifacts")
            continue
        print(f"[phase4_fix1] seed={seed} train start")
        rc = subprocess.run([sys.executable, str(train_script), "--seed", str(seed)]).returncode
        if rc != 0:
            raise RuntimeError(f"phase4_fix1 seed={seed} failed rc={rc}")

    # Tier1 eval (500 eps per seed).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_seeds = [int(x) for x in np.random.default_rng(2026).integers(0, 2**31 - 1, size=500)]
    per_seed = {}
    for seed in SEEDS:
        model = _load_ppo(MODEL_DIR / f"ppo_full_seed_{seed}.pt", device)
        rows = []
        for es in episode_seeds:
            rng = np.random.default_rng(es)
            rows.append(_rollout_episode(model, rng, device))
        _write_csv(EVAL_DIR / f"ppo_full_fix1_seed{seed}_T1_episodes.csv", rows)
        per_seed[seed] = rows

    # Summaries
    summary = {"per_seed": {}, "mean_over_seeds": {}}
    keys = ["DCR", "TCU", "CER", "inband_dosing_fraction"]
    for seed in SEEDS:
        vals = {k: np.asarray([r[k] for r in per_seed[seed]], dtype=np.float64) for k in keys}
        summary["per_seed"][str(seed)] = {f"{k}_mean": float(np.mean(vals[k])) for k in keys}
        summary["per_seed"][str(seed)].update({f"{k}_sd": float(np.std(vals[k], ddof=1)) for k in keys})

    # mean over seed means, and seed-level sd
    for k in keys:
        seed_means = np.asarray([summary["per_seed"][str(s)][f"{k}_mean"] for s in SEEDS], dtype=np.float64)
        summary["mean_over_seeds"][f"{k}_mean"] = float(np.mean(seed_means))
        summary["mean_over_seeds"][f"{k}_sd_over_seeds"] = float(np.std(seed_means, ddof=1))

    # PID reference from existing phase3 full eval (unchanged env settings)
    pid_path = Path("results/phase3_full/eval/pid_deadband_T1_episodes.csv")
    pid_rows = []
    with pid_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            pid_rows.append(
                {
                    "DCR": float(r["DCR"]),
                    "TCU": float(r["TCU"]),
                    "CER": float(r["CER"]),
                    # PID csv does not include in-band dosing fraction; compute via policy audit quickly
                }
            )
    summary["pid_reference_T1"] = {
        "DCR_mean": float(np.mean([r["DCR"] for r in pid_rows])),
        "TCU_mean": float(np.mean([r["TCU"] for r in pid_rows])),
        "CER_mean": float(np.mean([r["CER"] for r in pid_rows])),
    }

    # Compute PID in-band dosing fraction with same 500-episode seeds.
    pid_pol = p2.make_policy_pid(p2.ziegler_nichols_pid_gains())
    inband = 0
    inband_dose = 0
    for es in episode_seeds:
        rng = np.random.default_rng(es)
        env = p2.make_sim_env(enable_curriculum_masking=False, enable_escalation_penalty=True, A_T=1.5, C_T=1.0)
        env.reset(rng)
        for _ in range(m.T_MAX):
            ph = env.ph
            if m.PH_LO <= ph <= m.PH_HI:
                inband += 1
            a = int(pid_pol(env, rng))
            if (m.PH_LO <= ph <= m.PH_HI) and (a != 0):
                inband_dose += 1
            _, _, done, _ = env.step(a, rng)
            if done:
                break
    summary["pid_reference_T1"]["inband_dosing_fraction"] = float(inband_dose / max(1, inband))

    (OUT_ROOT / "phase4_fix1_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
