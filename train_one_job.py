#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import phase3_full_run as p3


def _paths(variant: str, seed: int) -> tuple[Path, Path]:
    if variant == "ddpg":
        return (
            p3.MODELS_DIR / "ddpg" / f"ddpg_seed_{seed}.pt",
            p3.CURVES_DIR / "ddpg" / f"ddpg_seed_{seed}_curve.csv",
        )
    return (
        p3.MODELS_DIR / variant / f"{variant}_seed_{seed}.pt",
        p3.CURVES_DIR / variant / f"{variant}_seed_{seed}_curve.csv",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one Phase-3 job in isolated subprocess.")
    ap.add_argument("--variant", required=True, choices=list(p3.PPO_VARIANTS.keys()) + ["ddpg"])
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    p3._ensure_dirs()
    variant = str(args.variant)
    seed = int(args.seed)
    model_path, curve_path = _paths(variant, seed)

    if model_path.exists() and curve_path.exists():
        print(f"[train_one_job] reuse existing artifacts variant={variant} seed={seed}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[train_one_job] start variant={variant} seed={seed} device={device} "
        f"locked(reward_scale={p3.LOCKED_REWARD_SCALE}, gamma={p3.LOCKED_GAMMA}, vf_coef={p3.LOCKED_VF_COEF})"
    )
    t0 = time.time()
    if variant == "ddpg":
        env_kwargs: Dict[str, bool] = {"enable_curriculum_masking": True, "enable_escalation_penalty": True}
        agent, rows = p3._train_ddpg(seed=seed, device=device, env_kwargs=env_kwargs, live_curve_path=curve_path)
        torch.save({"seed": seed, "state_dict": agent.state_dict()}, model_path)
        if not rows:
            raise RuntimeError("No DDPG rows produced")
    else:
        env_kwargs = dict(p3.PPO_VARIANTS[variant])
        policy, rows = p3._train_ppo_variant(
            seed=seed,
            device=device,
            variant_name=variant,
            env_kwargs=env_kwargs,
            live_curve_path=curve_path,
        )
        torch.save(
            {
                "seed": seed,
                "state_dict": policy.state_dict(),
                "obs_rms_mean": policy.obs_rms.mean.detach().cpu(),
                "obs_rms_var": policy.obs_rms.var.detach().cpu(),
                "obs_rms_count": float(policy.obs_rms.count),
            },
            model_path,
        )
        if not rows:
            raise RuntimeError("No PPO rows produced")

    final_dcr = p3._existing_final_dcr(curve_path)
    print(
        f"[train_one_job] done variant={variant} seed={seed} wall_s={time.time()-t0:.1f} "
        f"final_eval_dcr={final_dcr}"
    )
    (p3.OUT_ROOT / "last_train_one_job.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": seed,
                "model_path": str(model_path),
                "curve_path": str(curve_path),
                "final_eval_dcr": final_dcr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
