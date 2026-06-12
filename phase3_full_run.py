#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

import phase2_baselines_mechanism as p2
import water_experiments_small as wes
import water_methodology_impl as m


# Locked configuration (user-specified; do not alter).
SEEDS: Sequence[int] = (11, 22, 33)
LOCKED_REWARD_SCALE = 0.1
LOCKED_GAMMA = 0.99
LOCKED_VF_COEF = 0.5
LOCKED_PPO_ROLLOUT = 2048
LOCKED_PPO_MINIBATCH = 512
LOCKED_PPO_WARMUP = 10_000
LOCKED_PPO_TOTAL_STEPS = 5_000_000
LOCKED_PPO_EVAL_EVERY = 100_000
LOCKED_PPO_EVAL_EPISODES = 20
LOCKED_DDPG_TOTAL_STEPS = 5_000_000
LOCKED_DDPG_EVAL_EVERY = 5_000
LOCKED_DDPG_EVAL_EPISODES = 20
LOCKED_DDPG_BATCH = wes.SAMPLE_DD_PG_BATCH
LOCKED_DDPG_BUF = wes.SAMPLE_DD_PG_BUF

TIER1_EPISODES = 500
TIER2_EPISODES = 200

OUT_ROOT = Path("results/phase3_full")
MODELS_DIR = OUT_ROOT / "models"
CURVES_DIR = OUT_ROOT / "curves"
EVAL_DIR = OUT_ROOT / "eval"
STATS_DIR = OUT_ROOT / "stats"

PPO_VARIANTS: Dict[str, Dict[str, bool]] = {
    "ppo_full": {"enable_curriculum_masking": True, "enable_escalation_penalty": True},
    "ppo_no_curriculum": {"enable_curriculum_masking": False, "enable_escalation_penalty": True},
    "ppo_no_escalation": {"enable_curriculum_masking": True, "enable_escalation_penalty": False},
    "ppo_neither": {"enable_curriculum_masking": False, "enable_escalation_penalty": False},
}


@dataclass
class ModelArtifact:
    model_path: str
    curve_path: str
    final_eval_dcr: Optional[float]
    wall_seconds: float


def _ensure_dirs() -> None:
    for d in (OUT_ROOT, MODELS_DIR, CURVES_DIR, EVAL_DIR, STATS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for key in PPO_VARIANTS:
        (MODELS_DIR / key).mkdir(parents=True, exist_ok=True)
        (CURVES_DIR / key).mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "ddpg").mkdir(parents=True, exist_ok=True)
    (CURVES_DIR / "ddpg").mkdir(parents=True, exist_ok=True)


def _save_curve_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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


def _env_factory(
    *,
    enable_curriculum_masking: bool,
    enable_escalation_penalty: bool,
    A_T: float = 1.5,
    C_T: float = 1.0,
) -> m.WastewaterMDP:
    return p2.make_sim_env(
        enable_curriculum_masking=enable_curriculum_masking,
        enable_escalation_penalty=enable_escalation_penalty,
        A_T=A_T,
        C_T=C_T,
    )


def _evaluate_ppo_dcr(
    policy: m.ActorCritic,
    device: torch.device,
    episodes: int,
    seed: int,
    env_kwargs: Dict[str, bool],
) -> float:
    policy.eval()
    rng = np.random.default_rng(seed)
    dcrs = []
    for _ in range(episodes):
        env = _env_factory(**env_kwargs)
        obs = env.reset(rng).astype(np.float32)
        phs = [env.ph]
        for _ in range(m.T_MAX):
            with torch.no_grad():
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                dist, _ = policy(ot, torch.ones(1, 11, device=device))
                a = int(dist.sample().item())
            obs, _, done, _ = env.step(a, rng)
            phs.append(env.ph)
            if done:
                break
        arr = np.asarray(phs)
        dcrs.append(float(np.mean((arr >= m.PH_LO) & (arr <= m.PH_HI)) * 100.0))
    policy.train()
    return float(np.mean(dcrs))


def _train_ppo_variant(
    seed: int,
    device: torch.device,
    variant_name: str,
    env_kwargs: Dict[str, bool],
    live_curve_path: Optional[Path] = None,
) -> Tuple[m.ActorCritic, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = m.ActorCritic().to(device)
    policy.obs_rms.to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    env = _env_factory(**env_kwargs)
    obs = env.reset(rng).astype(np.float32)
    step_count = 0
    last_eval = 0
    rows: List[Dict[str, float]] = []
    live_fh = None
    live_writer = None
    if live_curve_path is not None:
        live_curve_path.parent.mkdir(parents=True, exist_ok=True)
        live_fh = live_curve_path.open("w", newline="", encoding="utf-8")
        live_writer = csv.DictWriter(
            live_fh,
            fieldnames=["step", "rollout_reward_mean", "rollout_reward_std", "eval_dcr_mean"],
        )
        live_writer.writeheader()
        live_fh.flush()

    def lr_of(s: int) -> float:
        if s < LOCKED_PPO_WARMUP:
            return 3e-4 * (s + 1) / max(1, LOCKED_PPO_WARMUP)
        prog = min(1.0, (s - LOCKED_PPO_WARMUP) / max(1, LOCKED_PPO_TOTAL_STEPS - LOCKED_PPO_WARMUP))
        return 3e-4 + (3e-5 - 3e-4) * 0.5 * (1 + np.cos(np.pi * prog))

    def ent_of(s: int) -> float:
        prog = min(1.0, s / max(1, m.ENT_DECAY_STEPS))
        return m.ENT_START + (m.ENT_END - m.ENT_START) * prog

    while step_count < LOCKED_PPO_TOTAL_STEPS:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(LOCKED_PPO_ROLLOUT):
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask(rng)).float().unsqueeze(0).to(device)
            dist, v = policy(ot, mask)
            a = dist.sample()
            logp = dist.log_prob(a)
            obs2, r, done, _ = env.step(int(a.item()), rng)
            policy.obs_rms.update(ot)

            obs_buf.append(obs.copy())
            act_buf.append(int(a.item()))
            logp_buf.append(float(logp.item()))
            rew_buf.append(float(r) * LOCKED_REWARD_SCALE)  # locked training-only scaling
            val_buf.append(float(v.item()))
            done_buf.append(float(done))

            obs = obs2.astype(np.float32)
            step_count += 1
            if done:
                obs = env.reset(rng).astype(np.float32)
            if step_count >= LOCKED_PPO_TOTAL_STEPS:
                break

        with torch.no_grad():
            ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            _, v_last = policy(ot, torch.ones(1, 11, device=device))
            last = float(v_last.item())

        rewards = np.asarray(rew_buf, dtype=np.float64)
        values = np.asarray(val_buf + [last], dtype=np.float64)
        dones = np.asarray(done_buf, dtype=np.float64)

        adv = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + LOCKED_GAMMA * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + LOCKED_GAMMA * m.GAE_LAMBDA * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:-1]

        obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        logp_old = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # locked

        n = obs_t.size(0)
        idx = np.arange(n)
        for _ in range(m.PPO_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, LOCKED_PPO_MINIBATCH):
                mb = idx[start : start + LOCKED_PPO_MINIBATCH]
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
                loss = -(surr.mean()) + LOCKED_VF_COEF * vf_loss - ent_of(step_count) * ent
                for g in opt.param_groups:
                    g["lr"] = float(lr_of(step_count))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), m.MAX_GRAD_NORM)
                opt.step()

        eval_dcr = np.nan
        if (step_count - last_eval) >= LOCKED_PPO_EVAL_EVERY or step_count >= LOCKED_PPO_TOTAL_STEPS:
            eval_dcr = _evaluate_ppo_dcr(policy, device, LOCKED_PPO_EVAL_EPISODES, seed + step_count, env_kwargs)
            last_eval = step_count

        row = {
            "step": float(step_count),
            "rollout_reward_mean": float(np.mean(rewards)),
            "rollout_reward_std": float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0),
            "eval_dcr_mean": float(eval_dcr) if not np.isnan(eval_dcr) else np.nan,
        }
        rows.append(row)
        if live_writer is not None and live_fh is not None:
            live_writer.writerow(row)
            live_fh.flush()
        if step_count % 250_000 == 0:
            print(f"[{variant_name}] seed={seed} step={step_count}/{LOCKED_PPO_TOTAL_STEPS}")

    if live_fh is not None:
        live_fh.close()
    return policy, rows


def _train_ddpg(
    seed: int,
    device: torch.device,
    env_kwargs: Dict[str, bool],
    live_curve_path: Optional[Path] = None,
) -> Tuple[wes.DDPGDiscrete, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = wes.DDPGDiscrete().to(device)
    opt_a = optim.Adam(agent.actor.parameters(), lr=1e-4)
    opt_c = optim.Adam(agent.critic.parameters(), lr=1e-3)
    buf: deque = deque(maxlen=LOCKED_DDPG_BUF)
    env = _env_factory(**env_kwargs)
    obs = env.reset(rng).astype(np.float32)
    rows: List[Dict[str, float]] = []
    last_eval = 0
    critic_recent: List[float] = []
    actor_recent: List[float] = []
    live_fh = None
    live_writer = None
    if live_curve_path is not None:
        live_curve_path.parent.mkdir(parents=True, exist_ok=True)
        live_fh = live_curve_path.open("w", newline="", encoding="utf-8")
        live_writer = csv.DictWriter(
            live_fh,
            fieldnames=["step", "critic_loss_mean_recent", "actor_loss_mean_recent", "eval_dcr_mean"],
        )
        live_writer.writeheader()
        live_fh.flush()

    for step in range(1, LOCKED_DDPG_TOTAL_STEPS + 1):
        with torch.no_grad():
            u = agent.actor(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            u = torch.clamp(u + 0.1 * torch.randn_like(u), -1, 1)
        a_idx = int(wes.DDPGDiscrete.u_to_action(u).item())
        obs2, r, done, _ = env.step(a_idx, rng)
        buf.append((obs.copy(), float(u.item()), float(r) * LOCKED_REWARD_SCALE, obs2.copy(), float(done)))
        obs = env.reset(rng).astype(np.float32) if done else obs2.astype(np.float32)

        if len(buf) >= LOCKED_DDPG_BATCH:
            idx = rng.choice(len(buf), size=LOCKED_DDPG_BATCH, replace=False)
            batch = [buf[i] for i in idx]
            s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
            u_b = torch.tensor([[b[1]] for b in batch], dtype=torch.float32, device=device)
            r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
            sp = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
            d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

            with torch.no_grad():
                up = agent.actor_t(sp)
                qn = agent.critic_t(torch.cat([sp, up], dim=1)).squeeze(-1)
                y = r_b + LOCKED_GAMMA * (1.0 - d_b) * qn

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
            critic_recent.append(float(loss_c.item()))
            actor_recent.append(float(loss_a.item()))

        eval_dcr = np.nan
        if (step - last_eval) >= LOCKED_DDPG_EVAL_EVERY or step >= LOCKED_DDPG_TOTAL_STEPS:
            eval_vals = wes.ddpg_dcr_scores(agent, lambda: _env_factory(**env_kwargs), LOCKED_DDPG_EVAL_EPISODES, device)
            eval_dcr = float(np.mean(eval_vals))
            last_eval = step

        row = {
            "step": float(step),
            "critic_loss_mean_recent": float(np.mean(critic_recent[-500:])) if critic_recent else np.nan,
            "actor_loss_mean_recent": float(np.mean(actor_recent[-500:])) if actor_recent else np.nan,
            "eval_dcr_mean": eval_dcr,
        }
        rows.append(row)
        if live_writer is not None and live_fh is not None:
            live_writer.writerow(row)
            live_fh.flush()
        if step % 500_000 == 0:
            print(f"[DDPG] seed={seed} step={step}/{LOCKED_DDPG_TOTAL_STEPS}")

    if live_fh is not None:
        live_fh.close()
    return agent, rows


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


def _load_ddpg(path: Path, device: torch.device) -> wes.DDPGDiscrete:
    ckpt = torch.load(path, map_location=device)
    model = wes.DDPGDiscrete().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _derive_tier2_shift_ranges() -> Dict[str, float]:
    ds2_path = Path("data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv")
    if not ds2_path.exists():
        return {"A_T_lo": 0.8, "A_T_hi": 3.0, "C_T_lo": 0.7, "C_T_hi": 2.2, "source": "fallback defaults"}

    df = pd.read_csv(ds2_path)
    cols = {c.lower(): c for c in df.columns}
    hard_col = None
    cond_col = None
    for c in df.columns:
        cl = c.lower()
        if hard_col is None and ("hardness" in cl or "hard" in cl):
            hard_col = c
        if cond_col is None and ("specific_conductance" in cl or "conduct" in cl or cl == "spc"):
            cond_col = c
    if hard_col is None or cond_col is None:
        return {"A_T_lo": 0.8, "A_T_hi": 3.0, "C_T_lo": 0.7, "C_T_hi": 2.2, "source": "fallback defaults (columns missing)"}

    hard = pd.to_numeric(df[hard_col], errors="coerce").dropna().to_numpy()
    cond = pd.to_numeric(df[cond_col], errors="coerce").dropna().to_numpy()
    if len(hard) < 10 or len(cond) < 10:
        return {"A_T_lo": 0.8, "A_T_hi": 3.0, "C_T_lo": 0.7, "C_T_hi": 2.2, "source": "fallback defaults (insufficient rows)"}

    qh_lo, qh_hi = np.quantile(hard, [0.1, 0.9])
    qc_lo, qc_hi = np.quantile(cond, [0.1, 0.9])
    # Eq-like mapping used in codebase for A_T.
    A_lo = float(np.clip(1.8 * (qh_lo / 100.0) + 2.2 * (qc_lo / 1000.0), 0.5, 12.0))
    A_hi = float(np.clip(1.8 * (qh_hi / 100.0) + 2.2 * (qc_hi / 1000.0), 0.5, 12.0))
    # C_T from conductivity + Henry term at 20C fixed reference.
    tk = 293.15
    kH = 0.033 * math.exp(2400.0 * (1.0 / 298.15 - 1.0 / tk))
    C_lo = float(np.clip(0.018 * (qc_lo / 1000.0) + 11.0 * kH, 0.5, 25.0))
    C_hi = float(np.clip(0.018 * (qc_hi / 1000.0) + 11.0 * kH, 0.5, 25.0))
    if A_lo > A_hi:
        A_lo, A_hi = A_hi, A_lo
    if C_lo > C_hi:
        C_lo, C_hi = C_hi, C_lo
    return {
        "A_T_lo": A_lo,
        "A_T_hi": A_hi,
        "C_T_lo": C_lo,
        "C_T_hi": C_hi,
        "source": f"derived from real DS-2 quantiles ({hard_col}, {cond_col})",
    }


def _episode_metrics(ph: np.ndarray, acts: np.ndarray) -> Dict[str, float]:
    dcr = float(np.mean((ph >= m.PH_LO) & (ph <= m.PH_HI)) * 100.0)
    mpd = float(np.mean(np.abs(ph - m.PH_MID)))
    doses = np.asarray([m.ACTION_VOLUMES_ML[int(a)] for a in acts], dtype=np.float64)
    tcu = float(np.sum(doses))
    cer = float(dcr / (1.0 + tcu))
    d1 = np.diff(ph)
    oec = float(np.sum((np.sign(d1[1:]) != np.sign(d1[:-1])) & (np.abs(d1[1:]) > 0.2))) if len(d1) > 1 else 0.0
    first_stab = m.T_MAX
    inband = ((ph >= m.PH_LO) & (ph <= m.PH_HI)).astype(np.float64)
    for t in range(7, len(inband)):
        if np.all(inband[t - 7 : t + 1] > 0.5):
            first_stab = t
            break
    pdcr = float(np.mean((acts <= 7).astype(np.float64)) if len(acts) else 0.0)
    return {
        "DCR": dcr,
        "MPD": mpd,
        "TCU": tcu,
        "CER": cer,
        "OEC": oec,
        "PST": float(first_stab),
        "PDCR": pdcr,
        "STG": float("nan"),
    }


def _rollout_controller(
    controller: str,
    rng: np.random.Generator,
    tier: str,
    tier2_ranges: Dict[str, float],
    model: Optional[object] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    if tier == "T1":
        A_T, C_T = 1.5, 1.0
    else:
        A_T = float(rng.uniform(tier2_ranges["A_T_lo"], tier2_ranges["A_T_hi"]))
        C_T = float(rng.uniform(tier2_ranges["C_T_lo"], tier2_ranges["C_T_hi"]))
    env = _env_factory(enable_curriculum_masking=False, enable_escalation_penalty=True, A_T=A_T, C_T=C_T)
    obs = env.reset(rng).astype(np.float32)
    phs = [env.ph]
    acts: List[int] = []
    pid_ctrl = p2.make_policy_pid(p2.ziegler_nichols_pid_gains()) if controller == "pid" else None
    lut = p2.build_static_lookup_table()
    for _ in range(m.T_MAX):
        if controller == "rule":
            a = p2.policy_rule_based(env, rng)
        elif controller == "pid":
            a = pid_ctrl(env, rng)  # type: ignore[operator]
        elif controller == "lut":
            a = p2.lookup_table_action(env.ph, lut)
        elif controller == "null":
            a = 0
        elif controller == "random":
            a = int(rng.integers(0, 11))
        elif controller == "ppo":
            assert model is not None and device is not None
            with torch.no_grad():
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                dist, _ = model(ot, torch.ones(1, 11, device=device))
                a = int(dist.sample().item())
        elif controller == "ddpg":
            assert model is not None and device is not None
            with torch.no_grad():
                u = model.actor(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                a = int(wes.DDPGDiscrete.u_to_action(u).item())
        else:
            raise ValueError(controller)
        obs, _, done, _ = env.step(int(a), rng)
        acts.append(int(a))
        phs.append(env.ph)
        if done:
            break
    return _episode_metrics(np.asarray(phs, dtype=np.float64), np.asarray(acts, dtype=np.int64))


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(d) / sd)


def _bootstrap_ci_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (x[idx] - y[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _wilcoxon_with_bonferroni(x: np.ndarray, y: np.ndarray, m_tests: int) -> Dict[str, float]:
    stat, p = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    p_adj = min(float(p) * m_tests, 1.0)
    return {"stat": float(stat), "p": float(p), "p_bonf": p_adj}


def _aggregate_episode_means(per_seed_rows: Dict[int, List[Dict[str, float]]]) -> List[Dict[str, float]]:
    episodes = len(next(iter(per_seed_rows.values())))
    out = []
    for i in range(episodes):
        row: Dict[str, float] = {}
        for k in per_seed_rows[next(iter(per_seed_rows))][i].keys():
            vals = [per_seed_rows[s][i][k] for s in per_seed_rows.keys()]
            row[k] = float(np.nanmean(vals))
        out.append(row)
    return out


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_full_phase3() -> None:
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_summary: Dict[str, object] = {
        "device": str(device),
        "locked_config": {
            "reward_scale": LOCKED_REWARD_SCALE,
            "gamma": LOCKED_GAMMA,
            "vf_coef": LOCKED_VF_COEF,
            "critic": "default",
            "advantage_norm": True,
            "reward_form": "raw with signed compliance term, W_COMP=3.0",
        },
        "training": {},
    }
    # 3a: Sequential subprocess isolation (one fresh process per job).
    train_jobs: List[Tuple[str, int]] = []
    for variant_name in PPO_VARIANTS.keys():
        run_summary["training"][variant_name] = {}
        for seed in SEEDS:
            train_jobs.append((variant_name, int(seed)))
    run_summary["training"]["ddpg"] = {}
    for seed in SEEDS:
        train_jobs.append(("ddpg", int(seed)))

    pending_jobs: List[Tuple[str, int]] = []
    for variant_name, seed in train_jobs:
        if variant_name == "ddpg":
            model_path = MODELS_DIR / "ddpg" / f"ddpg_seed_{seed}.pt"
            curve_path = CURVES_DIR / "ddpg" / f"ddpg_seed_{seed}_curve.csv"
        else:
            model_path = MODELS_DIR / variant_name / f"{variant_name}_seed_{seed}.pt"
            curve_path = CURVES_DIR / variant_name / f"{variant_name}_seed_{seed}_curve.csv"
        if model_path.exists() and curve_path.exists():
            final_dcr = _existing_final_dcr(curve_path)
            run_summary["training"][variant_name][str(seed)] = ModelArtifact(
                model_path=str(model_path),
                curve_path=str(curve_path),
                final_eval_dcr=final_dcr,
                wall_seconds=0.0,
            ).__dict__
            print(f"[{variant_name}] seed={seed} reuse existing artifacts")
        else:
            pending_jobs.append((variant_name, seed))

    print(f"[TRAIN_QUEUE] total_jobs={len(train_jobs)} pending_jobs={len(pending_jobs)} workers=1-subprocess")
    train_one_job = str((Path(__file__).resolve().parent / "train_one_job.py"))
    for variant_name, seed in pending_jobs:
        t0 = time.time()
        print(f"[TRAIN_START] variant={variant_name} seed={seed}")
        cp = subprocess.run(
            [sys.executable, train_one_job, "--variant", variant_name, "--seed", str(seed)],
            check=False,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"Training subprocess failed: variant={variant_name} seed={seed} rc={cp.returncode}")
        if variant_name == "ddpg":
            model_path = MODELS_DIR / "ddpg" / f"ddpg_seed_{seed}.pt"
            curve_path = CURVES_DIR / "ddpg" / f"ddpg_seed_{seed}_curve.csv"
        else:
            model_path = MODELS_DIR / variant_name / f"{variant_name}_seed_{seed}.pt"
            curve_path = CURVES_DIR / variant_name / f"{variant_name}_seed_{seed}_curve.csv"
        if not (model_path.exists() and curve_path.exists()):
            raise RuntimeError(f"Missing artifacts after subprocess success: variant={variant_name} seed={seed}")
        final_dcr = _existing_final_dcr(curve_path)
        run_summary["training"][variant_name][str(seed)] = {
            "model_path": str(model_path),
            "curve_path": str(curve_path),
            "final_eval_dcr": final_dcr,
            "wall_seconds": float(time.time() - t0),
            "status": "trained_subprocess",
        }
        print(f"[TRAIN_DONE] variant={variant_name} seed={seed} final_eval_dcr={final_dcr}")

    (OUT_ROOT / "phase3a_training_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # 3b: Evaluation
    tier2_ranges = _derive_tier2_shift_ranges()
    eval_summary: Dict[str, object] = {
        "tier1_episodes": TIER1_EPISODES,
        "tier2_episodes": TIER2_EPISODES,
        "tier2_shift_ranges": tier2_ranges,
        "fairness": "All controllers evaluated in identical simulator dynamics and metric computation. "
        "Training reward scaling (0.1) used only during learning updates and not in evaluation metrics.",
    }

    tier_episode_seeds = {
        "T1": [int(x) for x in np.random.default_rng(2026).integers(0, 2**31 - 1, size=TIER1_EPISODES)],
        "T2": [int(x) for x in np.random.default_rng(2027).integers(0, 2**31 - 1, size=TIER2_EPISODES)],
    }

    # Load learned models
    ppo_models: Dict[str, Dict[int, m.ActorCritic]] = {}
    for variant_name in PPO_VARIANTS.keys():
        ppo_models[variant_name] = {}
        for seed in SEEDS:
            p = MODELS_DIR / variant_name / f"{variant_name}_seed_{seed}.pt"
            ppo_models[variant_name][seed] = _load_ppo(p, device)
    ddpg_models: Dict[int, wes.DDPGDiscrete] = {
        seed: _load_ddpg(MODELS_DIR / "ddpg" / f"ddpg_seed_{seed}.pt", device) for seed in SEEDS
    }

    all_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = {"T1": {}, "T2": {}}

    def eval_deterministic_controller(name: str, ctrl_key: str, tier: str):
        rows = []
        for ep_seed in tier_episode_seeds[tier]:
            rng = np.random.default_rng(ep_seed)
            mrow = _rollout_controller(ctrl_key, rng, tier, tier2_ranges)
            rows.append(mrow)
        all_metrics[tier][name] = rows
        _write_csv(EVAL_DIR / f"{name}_{tier}_episodes.csv", rows)

    def eval_learned_ppo(name: str, variant_name: str, tier: str):
        per_seed_rows: Dict[int, List[Dict[str, float]]] = {}
        for seed in SEEDS:
            rows = []
            for ep_seed in tier_episode_seeds[tier]:
                rng = np.random.default_rng(ep_seed)
                mrow = _rollout_controller("ppo", rng, tier, tier2_ranges, model=ppo_models[variant_name][seed], device=device)
                rows.append(mrow)
            per_seed_rows[seed] = rows
            _write_csv(EVAL_DIR / f"{name}_seed{seed}_{tier}_episodes.csv", rows)
        mean_rows = _aggregate_episode_means(per_seed_rows)
        all_metrics[tier][name] = mean_rows
        _write_csv(EVAL_DIR / f"{name}_{tier}_episodes_mean_over_seeds.csv", mean_rows)

    def eval_learned_ddpg(name: str, tier: str):
        per_seed_rows: Dict[int, List[Dict[str, float]]] = {}
        for seed in SEEDS:
            rows = []
            for ep_seed in tier_episode_seeds[tier]:
                rng = np.random.default_rng(ep_seed)
                mrow = _rollout_controller("ddpg", rng, tier, tier2_ranges, model=ddpg_models[seed], device=device)
                rows.append(mrow)
            per_seed_rows[seed] = rows
            _write_csv(EVAL_DIR / f"{name}_seed{seed}_{tier}_episodes.csv", rows)
        mean_rows = _aggregate_episode_means(per_seed_rows)
        all_metrics[tier][name] = mean_rows
        _write_csv(EVAL_DIR / f"{name}_{tier}_episodes_mean_over_seeds.csv", mean_rows)

    for tier in ("T1", "T2"):
        eval_deterministic_controller("rule_based", "rule", tier)
        eval_deterministic_controller("pid_deadband", "pid", tier)
        eval_deterministic_controller("lut", "lut", tier)
        eval_deterministic_controller("null", "null", tier)
        eval_deterministic_controller("random", "random", tier)
        eval_learned_ddpg("ddpg", tier)
        eval_learned_ppo("ppo_full", "ppo_full", tier)
        eval_learned_ppo("ppo_no_curriculum", "ppo_no_curriculum", tier)
        eval_learned_ppo("ppo_no_escalation", "ppo_no_escalation", tier)
        eval_learned_ppo("ppo_neither", "ppo_neither", tier)

    # Aggregate means/sd
    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {"T1": {}, "T2": {}}
    metric_keys = ["DCR", "MPD", "TCU", "CER", "OEC", "PST", "PDCR", "STG"]
    for tier in ("T1", "T2"):
        for ctrl, rows in all_metrics[tier].items():
            t: Dict[str, float] = {}
            for mk in metric_keys:
                vals = np.asarray([r[mk] for r in rows], dtype=np.float64)
                t[f"{mk}_mean"] = float(np.nanmean(vals))
                t[f"{mk}_sd"] = float(np.nanstd(vals, ddof=1))
            summary_table[tier][ctrl] = t
    (OUT_ROOT / "phase3b_evaluation_summary.json").write_text(json.dumps(summary_table, indent=2), encoding="utf-8")

    # 3c: Statistics
    stats_out: Dict[str, object] = {"tier1_ppo_vs_baselines": {}, "ablation_vs_full": {}}
    baselines = ["rule_based", "pid_deadband", "lut", "ddpg", "null", "random"]
    ppo_full_t1 = all_metrics["T1"]["ppo_full"]
    m_tests = len(baselines) * 3
    for b in baselines:
        stats_out["tier1_ppo_vs_baselines"][b] = {}
        for mk in ("DCR", "TCU", "CER"):
            x = np.asarray([r[mk] for r in ppo_full_t1], dtype=np.float64)
            y = np.asarray([r[mk] for r in all_metrics["T1"][b]], dtype=np.float64)
            w = _wilcoxon_with_bonferroni(x, y, m_tests)
            d = _cohens_d_paired(x, y)
            ci = _bootstrap_ci_diff(x, y, n_boot=5000, seed=123 + hash((b, mk)) % 10000)
            stats_out["tier1_ppo_vs_baselines"][b][mk] = {
                "wilcoxon_stat": w["stat"],
                "p": w["p"],
                "p_bonferroni": w["p_bonf"],
                "cohens_d_paired": d,
                "bootstrap95_diff_ci": [ci[0], ci[1]],
                "mean_diff": float(np.mean(x - y)),
            }

    ablations = ["ppo_no_curriculum", "ppo_no_escalation", "ppo_neither"]
    m_tests_ab = len(ablations) * 4
    for ab in ablations:
        stats_out["ablation_vs_full"][ab] = {}
        for mk in ("DCR", "TCU", "CER", "PDCR"):
            x = np.asarray([r[mk] for r in all_metrics["T1"]["ppo_full"]], dtype=np.float64)
            y = np.asarray([r[mk] for r in all_metrics["T1"][ab]], dtype=np.float64)
            w = _wilcoxon_with_bonferroni(x, y, m_tests_ab)
            d = _cohens_d_paired(x, y)
            ci = _bootstrap_ci_diff(x, y, n_boot=5000, seed=777 + hash((ab, mk)) % 10000)
            stats_out["ablation_vs_full"][ab][mk] = {
                "wilcoxon_stat": w["stat"],
                "p": w["p"],
                "p_bonferroni": w["p_bonf"],
                "cohens_d_paired": d,
                "bootstrap95_diff_ci": [ci[0], ci[1]],
                "mean_diff": float(np.mean(x - y)),
            }

    (OUT_ROOT / "phase3c_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    # 3d: actual_results.md
    def _hypothesis_verdict_h1() -> str:
        ppo = summary_table["T1"]["ppo_full"]
        best_dcr_base = max(summary_table["T1"][b]["DCR_mean"] for b in baselines)
        if ppo["DCR_mean"] > best_dcr_base and ppo["TCU_mean"] <= np.mean([summary_table["T1"][b]["TCU_mean"] for b in baselines]):
            return "SUPPORTED"
        if ppo["DCR_mean"] < best_dcr_base:
            return "NOT SUPPORTED"
        return "INCONCLUSIVE"

    def _hypothesis_verdict_h2() -> str:
        full = summary_table["T1"]["ppo_full"]
        better = 0
        worse = 0
        for ab in ablations:
            x = summary_table["T1"][ab]
            if full["CER_mean"] >= x["CER_mean"] and full["TCU_mean"] <= x["TCU_mean"] and full["PDCR_mean"] >= x["PDCR_mean"]:
                better += 1
            if full["DCR_mean"] < x["DCR_mean"]:
                worse += 1
        if better == len(ablations) and worse == 0:
            return "SUPPORTED"
        if better == 0:
            return "NOT SUPPORTED"
        return "INCONCLUSIVE"

    def _hypothesis_verdict_h3() -> str:
        p1 = summary_table["T1"]["ppo_full"]["DCR_mean"]
        p2 = summary_table["T2"]["ppo_full"]["DCR_mean"]
        if p2 < p1 - 5.0:
            return "SUPPORTED"
        if abs(p2 - p1) < 2.0:
            return "NOT SUPPORTED"
        return "INCONCLUSIVE"

    lines: List[str] = []
    lines.append("# Actual Results (Phase 3)")
    lines.append("")
    lines.append("All values below are from executed runs with locked configuration.")
    lines.append("")
    lines.append("## Locked Training Configuration")
    lines.append(f"- reward_scale (training only): `{LOCKED_REWARD_SCALE}`")
    lines.append(f"- gamma: `{LOCKED_GAMMA}`")
    lines.append(f"- vf_coef: `{LOCKED_VF_COEF}`")
    lines.append("- critic: `default`")
    lines.append("- advantage normalization: `on`")
    lines.append("- reward form: `raw reward with signed compliance term, W_COMP=3.0`")
    lines.append("")
    lines.append("## Fairness Note")
    lines.append(
        "Evaluation metrics are computed identically across all controllers from trajectory states/actions "
        "(DCR, MPD, TCU, CER, OEC, PST, PDCR, STG=NA). Training reward scaling is not used in metric computation."
    )
    lines.append("")
    for tier in ("T1", "T2"):
        lines.append(f"## {tier} Metrics (mean ± SD)")
        for ctrl, vals in summary_table[tier].items():
            lines.append(
                f"- {ctrl}: "
                f"DCR {vals['DCR_mean']:.2f}±{vals['DCR_sd']:.2f}, "
                f"MPD {vals['MPD_mean']:.4f}±{vals['MPD_sd']:.4f}, "
                f"TCU {vals['TCU_mean']:.2f}±{vals['TCU_sd']:.2f}, "
                f"CER {vals['CER_mean']:.4f}±{vals['CER_sd']:.4f}, "
                f"OEC {vals['OEC_mean']:.2f}±{vals['OEC_sd']:.2f}, "
                f"PST {vals['PST_mean']:.2f}±{vals['PST_sd']:.2f}, "
                f"PDCR {vals['PDCR_mean']:.4f}±{vals['PDCR_sd']:.4f}, "
                f"STG NA"
            )
        lines.append("")

    lines.append("## Tier 1 Statistics: PPO(full) vs Baselines (DCR, TCU, CER)")
    for b, bstats in stats_out["tier1_ppo_vs_baselines"].items():
        for mk in ("DCR", "TCU", "CER"):
            s = bstats[mk]
            lines.append(
                f"- PPO(full) vs {b} [{mk}]: Wilcoxon stat={s['wilcoxon_stat']:.3f}, "
                f"p={s['p']:.4g}, p_bonf={s['p_bonferroni']:.4g}, "
                f"Cohen's d={s['cohens_d_paired']:.4f}, "
                f"bootstrap95 diff CI=[{s['bootstrap95_diff_ci'][0]:.4f}, {s['bootstrap95_diff_ci'][1]:.4f}]"
            )
    lines.append("")

    lines.append("## Ablation Statistics: PPO(full) vs Reduced")
    for ab, astats in stats_out["ablation_vs_full"].items():
        for mk in ("DCR", "TCU", "CER", "PDCR"):
            s = astats[mk]
            lines.append(
                f"- PPO(full) vs {ab} [{mk}]: Wilcoxon stat={s['wilcoxon_stat']:.3f}, "
                f"p={s['p']:.4g}, p_bonf={s['p_bonferroni']:.4g}, "
                f"Cohen's d={s['cohens_d_paired']:.4f}, "
                f"bootstrap95 diff CI=[{s['bootstrap95_diff_ci'][0]:.4f}, {s['bootstrap95_diff_ci'][1]:.4f}]"
            )
    lines.append("")

    h1 = _hypothesis_verdict_h1()
    h2 = _hypothesis_verdict_h2()
    h3 = _hypothesis_verdict_h3()
    h4 = "DEFERRED (no public real validation dataset with suitable pH+timestamp structure)"
    lines.append("## Hypotheses")
    lines.append(f"- H1 (PPO vs baselines on DCR+TCU): **{h1}**")
    lines.append(f"- H2 (progressive dosing helps CER/TCU/PDCR without hurting DCR): **{h2}**")
    lines.append(f"- H3 (Tier 1 vs Tier 2 performance shift): **{h3}**")
    lines.append(f"- H4: **{h4}**")
    lines.append("")

    best_baseline = max(baselines, key=lambda b: summary_table["T1"][b]["DCR_mean"])
    if summary_table["T1"]["ppo_full"]["DCR_mean"] < summary_table["T1"][best_baseline]["DCR_mean"]:
        lines.append(
            f"PPO(full) does not beat the best baseline on Tier 1 DCR "
            f"({summary_table['T1']['ppo_full']['DCR_mean']:.2f}% vs {best_baseline} "
            f"{summary_table['T1'][best_baseline]['DCR_mean']:.2f}%)."
        )
        lines.append("")

    lines.append("## Threats to Validity")
    lines.append("- Simulation-only framing; no public empirical validation dataset met requirements.")
    lines.append("- Single environment model assumptions (titration dynamics, noise model).")
    lines.append("- Statistical conclusions depend on episode sampling distribution and OOD shift definition.")
    lines.append("")

    lines.append("## Manuscript Fill Values")
    lines.append("- Tier 1/Tier 2 metrics for each controller are in `phase3b_evaluation_summary.json`.")
    lines.append("- Statistical test outputs are in `phase3c_stats.json`.")
    lines.append("- Locked config and run provenance are in `phase3a_training_summary.json`.")

    Path("actual_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[DONE] wrote actual_results.md")


if __name__ == "__main__":
    run_full_phase3()
