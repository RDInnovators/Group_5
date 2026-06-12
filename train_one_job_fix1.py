#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import phase3_full_run as p3


OUT_ROOT = Path("results/phase4_fix1")
MODEL_DIR = OUT_ROOT / "models" / "ppo_full"
CURVE_DIR = OUT_ROOT / "curves" / "ppo_full"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one PPO-full seed for phase4 fix1.")
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    seed = int(args.seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"ppo_full_seed_{seed}.pt"
    curve_path = CURVE_DIR / f"ppo_full_seed_{seed}_curve.csv"

    if model_path.exists() and curve_path.exists():
        print(f"[fix1_train_one] reuse seed={seed}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_kwargs = dict(p3.PPO_VARIANTS["ppo_full"])
    print(
        f"[fix1_train_one] start seed={seed} device={device} "
        f"locked(reward_scale={p3.LOCKED_REWARD_SCALE}, gamma={p3.LOCKED_GAMMA}, vf_coef={p3.LOCKED_VF_COEF})"
    )
    t0 = time.time()
    policy, rows = p3._train_ppo_variant(
        seed=seed,
        device=device,
        variant_name="ppo_full",
        env_kwargs=env_kwargs,
        live_curve_path=curve_path,
    )
    if not rows:
        raise RuntimeError("No PPO rows produced for fix1 seed run")
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
    final_dcr = p3._existing_final_dcr(curve_path)
    print(f"[fix1_train_one] done seed={seed} wall_s={time.time()-t0:.1f} final_eval_dcr={final_dcr}")
    (OUT_ROOT / "last_fix1_train_one.json").write_text(
        json.dumps(
            {
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
