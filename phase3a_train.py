#!/usr/bin/env python3
"""
Phase 3a runner:
- Train PPO across 3 seeds at documented full budget.
- Train DDPG across same 3 seeds.
- Save models and training curves for later loading.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import phase2_baselines_mechanism as p2
import water_experiments_small as wes
import water_methodology_impl as m


SEEDS: Sequence[int] = (11, 22, 33)
OUT_ROOT = Path("results/phase3a")
PPO_MODEL_DIR = OUT_ROOT / "models" / "ppo"
DDPG_MODEL_DIR = OUT_ROOT / "models" / "ddpg"
CURVE_DIR = OUT_ROOT / "curves"

# Documented full-budget PPO setting from the codebase (demo_mode=False path).
PPO_TOTAL_STEPS = 5_000_000
PPO_ROLLOUT_LEN = 2048
PPO_MINIBATCH = 512
PPO_WARMUP = 10_000
PPO_EVAL_EVERY = 100_000
PPO_EVAL_EPISODES = 20

# Phase 3a full-budget DDPG setting for parity with PPO budget.
DDPG_TOTAL_STEPS = 5_000_000
DDPG_EVAL_EVERY = 2_500
DDPG_EVAL_EPISODES = 20
DDPG_BATCH = wes.SAMPLE_DD_PG_BATCH
DDPG_BUF = wes.SAMPLE_DD_PG_BUF


@dataclass
class TrainArtifacts:
    model_path: str
    curve_path: str
    wall_seconds: float
    final_eval_dcr: Optional[float]


def ensure_dirs() -> None:
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DDPG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)


def env_factory() -> m.WastewaterMDP:
    return p2.make_sim_env(enable_curriculum_masking=True, enable_escalation_penalty=True)


def evaluate_ppo_dcr(
    policy: m.ActorCritic,
    device: torch.device,
    episodes: int,
    seed: int,
) -> float:
    policy.eval()
    rng = np.random.default_rng(seed)
    dcrs: List[float] = []
    for _ in range(episodes):
        env = env_factory()
        obs = env.reset(rng).astype(np.float32)
        phs = [env.ph]
        for _t in range(m.T_MAX):
            with torch.no_grad():
                o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                dist, _ = policy(o, torch.ones(1, 11, device=device))
                a = int(dist.sample().item())
            obs, _, done, _ = env.step(a, rng)
            phs.append(env.ph)
            if done:
                break
        ph_arr = np.asarray(phs)
        dcrs.append(float(np.mean((ph_arr >= m.PH_LO) & (ph_arr <= m.PH_HI)) * 100.0))
    policy.train()
    return float(np.mean(dcrs))


def save_curve_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _existing_final_dcr(curve_path: Path) -> Optional[float]:
    if not curve_path.exists():
        return None
    final = None
    with curve_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row.get("eval_dcr_mean", "")
            if val and val.lower() != "nan":
                final = float(val)
    return final


def train_ppo_logged(seed: int, device: torch.device) -> Tuple[m.ActorCritic, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = m.ActorCritic().to(device)
    policy.obs_rms.to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    env = env_factory()
    obs = env.reset(rng).astype(np.float32)
    step_count = 0
    last_eval = 0
    curve_rows: List[Dict[str, float]] = []

    def lr_of(s: int) -> float:
        if s < PPO_WARMUP:
            return 3e-4 * (s + 1) / max(1, PPO_WARMUP)
        prog = min(1.0, (s - PPO_WARMUP) / max(1, PPO_TOTAL_STEPS - PPO_WARMUP))
        return 3e-4 + (3e-5 - 3e-4) * 0.5 * (1 + np.cos(np.pi * prog))

    def ent_of(s: int) -> float:
        prog = min(1.0, s / max(1, m.ENT_DECAY_STEPS))
        return m.ENT_START + (m.ENT_END - m.ENT_START) * prog

    while step_count < PPO_TOTAL_STEPS:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(PPO_ROLLOUT_LEN):
            o_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask(rng)).float().unsqueeze(0).to(device)
            dist, v = policy(o_t, mask)
            a = dist.sample()
            logp = dist.log_prob(a)
            obs2, r, done, _ = env.step(int(a.item()), rng)
            policy.obs_rms.update(o_t)
            obs_buf.append(obs.copy())
            act_buf.append(int(a.item()))
            logp_buf.append(float(logp.item()))
            rew_buf.append(float(r))
            val_buf.append(float(v.item()))
            done_buf.append(float(done))
            obs = obs2.astype(np.float32)
            step_count += 1
            if done:
                obs = env.reset(rng).astype(np.float32)
            if step_count >= PPO_TOTAL_STEPS:
                break

        with torch.no_grad():
            o_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            _, v_last = policy(o_t, torch.ones(1, 11, device=device))
            last = float(v_last.item())

        rewards = np.asarray(rew_buf, dtype=np.float64)
        values = np.asarray(val_buf + [last], dtype=np.float64)
        dones = np.asarray(done_buf, dtype=np.float64)
        adv = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + m.GAMMA * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + m.GAMMA * m.GAE_LAMBDA * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:-1]

        obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        logp_old = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs_t.size(0)
        idx = np.arange(n)
        for _ in range(m.PPO_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, PPO_MINIBATCH):
                mb = idx[start : start + PPO_MINIBATCH]
                if mb.size < 2:
                    continue
                ob = obs_t[mb]
                ac = act_t[mb]
                dist, v = policy(ob, None)
                logp = dist.log_prob(ac)
                ratio = torch.exp(logp - logp_old[mb])
                clip_adv = torch.clamp(ratio, 1 - m.PPO_CLIP, 1 + m.PPO_CLIP) * adv_t[mb]
                surr = torch.min(ratio * adv_t[mb], clip_adv)
                vf_loss = 0.5 * torch.mean((ret_t[mb] - v) ** 2)
                ent = dist.entropy().mean()
                loss = -(surr.mean()) + 0.5 * vf_loss - ent_of(step_count) * ent
                for g in opt.param_groups:
                    g["lr"] = float(lr_of(step_count))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), m.MAX_GRAD_NORM)
                opt.step()

        eval_dcr = None
        if (step_count - last_eval) >= PPO_EVAL_EVERY or step_count >= PPO_TOTAL_STEPS:
            eval_dcr = evaluate_ppo_dcr(policy, device, PPO_EVAL_EPISODES, seed + step_count)
            last_eval = step_count
        curve_rows.append(
            {
                "step": float(step_count),
                "rollout_reward_mean": float(np.mean(rewards) if len(rewards) else np.nan),
                "rollout_reward_std": float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0),
                "eval_dcr_mean": float(eval_dcr) if eval_dcr is not None else np.nan,
            }
        )

    return policy, curve_rows


def train_ddpg_logged(seed: int, device: torch.device) -> Tuple[wes.DDPGDiscrete, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = wes.DDPGDiscrete().to(device)
    opt_a = optim.Adam(agent.actor.parameters(), lr=1e-4)
    opt_c = optim.Adam(agent.critic.parameters(), lr=1e-3)
    buf: deque = deque(maxlen=DDPG_BUF)
    env = env_factory()
    obs = env.reset(rng).astype(np.float32)
    gamma = m.GAMMA
    curve_rows: List[Dict[str, float]] = []

    step_in_ep = 0
    ep_return = 0.0
    last_eval = 0
    critic_loss_recent: List[float] = []
    actor_loss_recent: List[float] = []

    for step in range(1, DDPG_TOTAL_STEPS + 1):
        with torch.no_grad():
            u = agent.actor(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            u = torch.clamp(u + 0.1 * torch.randn_like(u), -1, 1)
        a_idx = int(wes.DDPGDiscrete.u_to_action(u).item())
        obs2, r, done, _ = env.step(a_idx, rng)
        buf.append((obs.copy(), float(u.item()), float(r), obs2.copy(), float(done)))
        obs = env.reset(rng).astype(np.float32) if done else obs2.astype(np.float32)

        ep_return += float(r)
        step_in_ep += 1
        if done or step_in_ep >= m.T_MAX:
            ep_return = 0.0
            step_in_ep = 0

        if len(buf) >= DDPG_BATCH:
            idx = rng.choice(len(buf), size=DDPG_BATCH, replace=False)
            batch = [buf[i] for i in idx]
            s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
            u_b = torch.tensor([[b[1]] for b in batch], dtype=torch.float32, device=device)
            r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
            sp = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
            d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

            with torch.no_grad():
                up = agent.actor_t(sp)
                qn = agent.critic_t(torch.cat([sp, up], dim=1)).squeeze(-1)
                y = r_b + gamma * (1.0 - d_b) * qn

            q = agent.critic(torch.cat([s, u_b], dim=1)).squeeze(-1)
            loss_c = nn.functional.mse_loss(q, y)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            u_pred = agent.actor(s)
            loss_a = -agent.critic(torch.cat([s, u_pred], dim=1)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

            agent.soft_update(0.01)
            critic_loss_recent.append(float(loss_c.item()))
            actor_loss_recent.append(float(loss_a.item()))

        eval_dcr = np.nan
        if (step - last_eval) >= DDPG_EVAL_EVERY or step >= DDPG_TOTAL_STEPS:
            eval_vals = wes.ddpg_dcr_scores(agent, env_factory, DDPG_EVAL_EPISODES, device)
            eval_dcr = float(np.mean(eval_vals))
            last_eval = step

        curve_rows.append(
            {
                "step": float(step),
                "critic_loss_mean_recent": float(np.mean(critic_loss_recent[-500:])) if critic_loss_recent else np.nan,
                "actor_loss_mean_recent": float(np.mean(actor_loss_recent[-500:])) if actor_loss_recent else np.nan,
                "eval_dcr_mean": eval_dcr,
            }
        )

    return agent, curve_rows


def run_phase3a() -> Dict[str, object]:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: Dict[str, object] = {
        "device": str(device),
        "seeds": list(SEEDS),
        "budgets": {"ppo_total_steps": PPO_TOTAL_STEPS, "ddpg_total_steps": DDPG_TOTAL_STEPS},
        "ppo": {},
        "ddpg": {},
    }

    for seed in SEEDS:
        model_path = PPO_MODEL_DIR / f"ppo_seed_{seed}.pt"
        curve_path = CURVE_DIR / f"ppo_seed_{seed}_curve.csv"
        if model_path.exists() and curve_path.exists():
            final_dcr = _existing_final_dcr(curve_path)
            summary["ppo"][str(seed)] = TrainArtifacts(
                model_path=str(model_path),
                curve_path=str(curve_path),
                wall_seconds=0.0,
                final_eval_dcr=final_dcr,
            ).__dict__
            print(f"[PPO] seed={seed} already exists; reusing artifacts final_eval_dcr={final_dcr}")
            continue

        t0 = time.time()
        policy, curve = train_ppo_logged(seed, device)
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
        save_curve_csv(curve_path, curve, ["step", "rollout_reward_mean", "rollout_reward_std", "eval_dcr_mean"])
        final_dcr = next((float(r["eval_dcr_mean"]) for r in reversed(curve) if not np.isnan(r["eval_dcr_mean"])), None)
        summary["ppo"][str(seed)] = TrainArtifacts(
            model_path=str(model_path),
            curve_path=str(curve_path),
            wall_seconds=float(time.time() - t0),
            final_eval_dcr=final_dcr,
        ).__dict__
        print(f"[PPO] seed={seed} done in {time.time()-t0:.1f}s final_eval_dcr={final_dcr}")

    for seed in SEEDS:
        model_path = DDPG_MODEL_DIR / f"ddpg_seed_{seed}.pt"
        curve_path = CURVE_DIR / f"ddpg_seed_{seed}_curve.csv"
        if model_path.exists() and curve_path.exists():
            final_dcr = _existing_final_dcr(curve_path)
            summary["ddpg"][str(seed)] = TrainArtifacts(
                model_path=str(model_path),
                curve_path=str(curve_path),
                wall_seconds=0.0,
                final_eval_dcr=final_dcr,
            ).__dict__
            print(f"[DDPG] seed={seed} already exists; reusing artifacts final_eval_dcr={final_dcr}")
            continue

        t0 = time.time()
        agent, curve = train_ddpg_logged(seed, device)
        torch.save({"seed": seed, "state_dict": agent.state_dict()}, model_path)
        save_curve_csv(curve_path, curve, ["step", "critic_loss_mean_recent", "actor_loss_mean_recent", "eval_dcr_mean"])
        final_dcr = next((float(r["eval_dcr_mean"]) for r in reversed(curve) if not np.isnan(r["eval_dcr_mean"])), None)
        summary["ddpg"][str(seed)] = TrainArtifacts(
            model_path=str(model_path),
            curve_path=str(curve_path),
            wall_seconds=float(time.time() - t0),
            final_eval_dcr=final_dcr,
        ).__dict__
        print(f"[DDPG] seed={seed} done in {time.time()-t0:.1f}s final_eval_dcr={final_dcr}")

    summary_path = OUT_ROOT / "phase3a_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] wrote {summary_path}")
    return summary


if __name__ == "__main__":
    run_phase3a()
