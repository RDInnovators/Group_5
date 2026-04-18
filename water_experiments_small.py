"""
Short-sample *first pass* for Methodology v.01 - runs after `run_full_pipeline`.

Uses small budgets (few episodes, short DS slices, one seed) to exercise:
Sec 2.3.2 one-step MAPE, Table 6-style surrogate RMSE + ECE gates, Table 16
extended metrics, Tier-1/2/3 smoke labels, LUT + DDPG baselines, Wilcoxon with
Bonferroni correction for four tests vs PPO and Cohen's d on paired DCR scores.

Usage:
 python water_experiments_small.py
 # or
 from water_experiments_small import methodology_first_pass_small
 methodology_first_pass_small(demo_mode=True)
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import water_methodology_impl as m

# ---------------------------------------------------------------------------
# Small-run defaults (tune upward for manuscript-scale experiments)
# ---------------------------------------------------------------------------

SAMPLE_SEC232_STEPS = 80
SAMPLE_TIER_EPISODES = 4
SAMPLE_DD_PG_STEPS = 2_500
SAMPLE_DD_PG_BATCH = 64
SAMPLE_DD_PG_BUF = 8_000
SAMPLE_LSTM_METRIC_BATCHES = 24


def lut_greedy_action(ph: float, A_T: float, C_T: float) -> int:
 """One-step LUT: minimise |pH_next − PH_MID| over discrete doses (Table 15-style lookup)."""
 best_a, best = 0, 1e9
 for a in range(11):
 ph2 = m.f_titration(ph, a, A_T, C_T)
 s = abs(ph2 - m.PH_MID)
 if s < best:
 best, best_a = s, a
 return best_a


def validate_simulator_sec232(
 ds5_hz,
 A_T: float,
 C_T: float,
 n_steps: int = SAMPLE_SEC232_STEPS,
 seed: int = 0,
) -> Dict[str, float]:
 """
 Sec 2.3.2 **diagnostic** - one-step ΔpH error between titration and successive observed pH on DS-5.

 The manuscript’s full protocol (50×120 min open-loop windows) is **not** implemented here; see
 ``Implementation_and_Remaining_Work.md``. **median |ΔpH_err|** is compared to ``TABLE6_GATE_SEC232_MEDIAN_DPH``.
 Relative MAPE can be large when actions are random vs natural drift; use median as the gate metric.
 """
 rng = np.random.default_rng(seed)
 ph = ds5_hz["pH"].to_numpy(dtype=np.float64)
 n = min(n_steps, len(ph) - 1)
 if n < 3:
 return {
 "MAPE_pct": float("nan"),
 "MAPE_pct_median": float("nan"),
 "median_abs_dph_err": float("nan"),
 "n": float(n),
 }
 errs_pct, abs_dph = [], []
 for i in range(n):
 a = int(rng.integers(0, 11))
 pred_dph = m.f_titration(ph[i], a, A_T, C_T) - ph[i]
 act_dph = ph[i + 1] - ph[i]
 abs_e = abs(pred_dph - act_dph)
 abs_dph.append(abs_e)
 # Symmetric floor avoids huge % when observed step ΔpH ≈ 0 (random action vs natural drift).
 denom = max(abs(act_dph), abs(pred_dph), 1e-4)
 errs_pct.append(min(abs_e / denom * 100.0, 10_000.0))
 return {
 "MAPE_pct": float(np.mean(errs_pct)),
 "MAPE_pct_median": float(np.median(errs_pct)),
 "median_abs_dph_err": float(np.median(abs_dph)),
 "n": float(n),
 }


def lstm_ds5_step_metrics(
 ds5_15m: pd.DataFrame,
 preprocess_stats: Dict[str, Any],
 lstm: m.LSTMSurrogate,
 dph_scaler,
 mm_cols: list,
 A_T: float,
 C_T: float,
 device: torch.device,
 L: int = m.L_SEQ,
 max_batches: int = 16,
) -> Dict[str, float]:
 """
 Table 6 gate 4 (proxy) - LSTM one-step ΔpH error on DS-5 downsampled through **train** scalers (OOD path).
 Uses CUSUM+inverse labels on DS-5 pH only for action alignment (same protocol as DS-1).
 """
 f = ds5_15m.copy()
 if "conductivity_uScm" not in f.columns:
 f["conductivity_uScm"] = 900.0
 if "DO_mgL" not in f.columns:
 f["DO_mgL"] = 6.0
 if "turbidity_NTU" not in f.columns:
 f["turbidity_NTU"] = 10.0
 trf, _ = m.preprocess_monitor(f, stats=preprocess_stats, fit=False)
 # SeqDS needs len(frame) >= L + 2; use margin for at least one mini-batch
 if len(trf) < L + 32:
 return {"median_abs_dph_err": float("nan"), "rmse_dph": float("nan"), "n": 0.0}
 ds_ph = pd.DataFrame({"pH": trf["pH_phys"].to_numpy(dtype=np.float64)})
 acts = m.reconstruct_actions(ds_ph, A_T, C_T)
 trf = m.attach_actions(trf, acts)
 ds_eval = m.SeqDS(trf, mm_cols, L)
 lstm.eval()
 loader = torch.utils.data.DataLoader(ds_eval, batch_size=m.LSTM_BATCH, shuffle=False)
 abs_e, se = [], []
 with torch.no_grad():
 for k, (xb, yb, phb) in enumerate(loader):
 if k >= max_batches:
 break
 xb, yb, phb = xb.to(device), yb.to(device), phb.to(device)
 pred_s = lstm(xb, dropout_active=False)
 dph_hat = torch.from_numpy(
 dph_scaler.inverse_transform(pred_s.cpu().numpy().reshape(-1, 1)).astype(np.float32)
 ).squeeze(-1).to(device)
 dph_t = yb
 abs_e.append(torch.abs(dph_hat - dph_t).detach().cpu().numpy())
 se.append(torch.mean((dph_hat - dph_t) ** 2).item())
 if not abs_e:
 return {"median_abs_dph_err": float("nan"), "rmse_dph": float("nan"), "n": 0.0}
 ae = np.concatenate(abs_e, axis=0)
 return {
 "median_abs_dph_err": float(np.median(ae)),
 "rmse_dph": float(np.sqrt(np.mean(se))) if se else float("nan"),
 "n": float(len(ae)),
 }


def lstm_val_rmse_and_ece(
 lstm: m.LSTMSurrogate,
 va_ds: m.SeqDS,
 dph_scaler,
 device: torch.device,
 max_batches: int = SAMPLE_LSTM_METRIC_BATCHES,
) -> Dict[str, float]:
 """Table 6 - validation RMSE on next pH and MAE on ΔpH residuals (primary calibration-style gate)."""
 lstm.eval()
 loader = torch.utils.data.DataLoader(va_ds, batch_size=m.LSTM_BATCH, shuffle=False)
 se, resids = [], []
 with torch.no_grad():
 for k, (xb, yb, phb) in enumerate(loader):
 if k >= max_batches:
 break
 xb, yb, phb = xb.to(device), yb.to(device), phb.to(device)
 pred_s = lstm(xb, dropout_active=False)
 dph_hat = torch.from_numpy(
 dph_scaler.inverse_transform(pred_s.cpu().numpy().reshape(-1, 1)).astype(np.float32)
 ).squeeze(-1).to(device)
 ph_next_hat = torch.clamp(phb + dph_hat, 0.0, 14.0)
 ph_next = torch.clamp(phb + yb, 0.0, 14.0)
 mse_b = torch.mean((ph_next_hat - ph_next) ** 2).item()
 if np.isfinite(mse_b):
 se.append(mse_b)
 dph_t = yb.cpu().numpy().ravel()
 dph_p = dph_hat.cpu().numpy().ravel()
 resids.extend((dph_p - dph_t).tolist())
 se_ok = [x for x in se if np.isfinite(x)]
 rmse = float(np.sqrt(np.mean(se_ok))) if se_ok else float("nan")
 resids_np = np.asarray(resids, dtype=np.float64)
 resids_np = resids_np[np.isfinite(resids_np)]
 mae_dph = float(np.mean(np.abs(resids_np))) if len(resids_np) else float("nan")
 return {
 "surrogate_val_RMSE_pH": rmse,
 "surrogate_MAE_dph_residual": mae_dph,
 "n_residuals": float(len(resids_np)),
 }


def table6_gates(
 rmse: float,
 mae_dph: float,
 median_abs_dph_err: float,
 rmse_max: Optional[float] = None,
 mae_dph_max: Optional[float] = None,
 median_dph_err_max: Optional[float] = None,
) -> Dict[str, Any]:
 """Pass/fail vs *Methodology v.01* Table 6 (defaults from ``water_methodology_impl``)."""
 rmax = m.TABLE6_GATE_RMSE if rmse_max is None else rmse_max
 mmax = m.TABLE6_GATE_MAE_DPH if mae_dph_max is None else mae_dph_max
 smax = m.TABLE6_GATE_SEC232_MEDIAN_DPH if median_dph_err_max is None else median_dph_err_max
 return {
 "gate_RMSE_pass": bool(rmse == rmse and rmse < rmax),
 "gate_MAE_dph_pass": bool(mae_dph == mae_dph and mae_dph < mmax),
 "gate_sec232_median_dph_pass": bool(
 median_abs_dph_err == median_abs_dph_err and median_abs_dph_err < smax
 ),
 "thresholds": {
 "rmse_max": rmax,
 "mae_dph_max": mmax,
 "median_dph_err_max": smax,
 },
 }


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
 d = x - y
 sd = float(np.std(d, ddof=1))
 if sd < 1e-12:
 return float("nan")
 return float(np.mean(d) / sd)


def wilcoxon_bonferroni_four(
 ppo: np.ndarray,
 rbt: np.ndarray,
 pid: np.ndarray,
 lut: np.ndarray,
 ddpg: np.ndarray,
 alpha: float = 0.05,
) -> Dict[str, Any]:
 """Four paired tests vs PPO; Bonferroni-corrected α/4 per test."""
 names = ["PPO_vs_RBT", "PPO_vs_PID", "PPO_vs_LUT", "PPO_vs_DDPG"]
 pairs = [(ppo, rbt), (ppo, pid), (ppo, lut), (ppo, ddpg)]
 alpha_b = alpha / 4.0
 out: Dict[str, Any] = {"alpha_family": alpha, "alpha_per_test": alpha_b, "tests": {}}
 for name, (a, b) in zip(names, pairs):
 stat, p = m.wilcoxon_report(a, b, alpha_b)
 out["tests"][name] = {"wilcoxon_stat": stat, "p_value": p, "cohens_d_paired": cohens_d_paired(a, b)}
 return out


def _fix_rollout_dcr(env_factory, ctrl: str, episodes: int) -> np.ndarray:
 """Episode mean DCR using full trajectories (matches `rollout_policy` logic)."""
 dcrs = []
 rng_master = np.random.default_rng(789)
 for i in range(episodes):
 env = env_factory()
 rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
 obs = env.reset(rng)
 phs = [env.ph]
 pid = m.PIDController() if ctrl == "pid" else None
 for _t in range(m.T_MAX):
 if ctrl == "rbt":
 a = m.rule_based_action(env.ph)
 elif ctrl == "pid":
 a = pid.action(env.ph)
 elif ctrl == "lut":
 a = lut_greedy_action(env.ph, env.A_T, env.C_T)
 else:
 a = 0
 obs, _, done, _ = env.step(a, rng)
 phs.append(env.ph)
 if done:
 break
 ph_arr = np.asarray(phs)
 dcrs.append(float(np.mean((ph_arr >= m.PH_LO) & (ph_arr <= m.PH_HI)) * 100.0))
 return np.asarray(dcrs, dtype=np.float64)


class DDPGDiscrete(nn.Module):
 """Continuous actor u∈[−1,1] → discretised action index; critic Q(s,u)."""

 def __init__(self, obs_dim: int = 13):
 super().__init__()
 self.actor = nn.Sequential(
 nn.Linear(obs_dim, 128),
 nn.ReLU(),
 nn.Linear(128, 64),
 nn.ReLU(),
 nn.Linear(64, 1),
 nn.Tanh(),
 )
 self.critic = nn.Sequential(
 nn.Linear(obs_dim + 1, 128),
 nn.ReLU(),
 nn.Linear(128, 64),
 nn.ReLU(),
 nn.Linear(64, 1),
 )
 self.critic_t = copy.deepcopy(self.critic)
 self.actor_t = copy.deepcopy(self.actor)
 for p in self.critic_t.parameters():
 p.requires_grad = False
 for p in self.actor_t.parameters():
 p.requires_grad = False

 @staticmethod
 def u_to_action(u: torch.Tensor) -> torch.Tensor:
 a = ((u + 1.0) * 0.5 * 10.0).round().clamp(0, 10).long()
 return a.squeeze(-1)

 def soft_update(self, tau: float = 0.005):
 for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
 pt.data.copy_(pt.data * (1 - tau) + p.data * tau)
 for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
 pt.data.copy_(pt.data * (1 - tau) + p.data * tau)


def train_ddpg_small(
 env_factory: Callable[[], m.WastewaterMDP],
 device: torch.device,
 steps: int = SAMPLE_DD_PG_STEPS,
 seed: int = 7,
) -> DDPGDiscrete:
 torch.manual_seed(seed)
 rng = np.random.default_rng(seed)
 agent = DDPGDiscrete().to(device)
 opt_a = optim.Adam(agent.actor.parameters(), lr=1e-4)
 opt_c = optim.Adam(agent.critic.parameters(), lr=1e-3)
 buf: deque = deque(maxlen=SAMPLE_DD_PG_BUF)
 env = env_factory()
 obs = env.reset(rng).astype(np.float32)
 gamma = m.GAMMA

 def push(s, u, r, sp, done):
 buf.append((s, u, r, sp, done))

 for step in range(steps):
 with torch.no_grad():
 u = agent.actor(torch.from_numpy(obs).float().unsqueeze(0).to(device))
 u = torch.clamp(u + 0.1 * torch.randn_like(u), -1, 1)
 a_idx = int(DDPGDiscrete.u_to_action(u).item())
 obs2, r, done, _ = env.step(a_idx, rng)
 u_val = float(u.item())
 push(obs.copy(), u_val, r, obs2.copy(), done)
 obs = env.reset(rng).astype(np.float32) if done else obs2.astype(np.float32)

 if len(buf) < SAMPLE_DD_PG_BATCH:
 continue
 batch = [buf[i] for i in rng.choice(len(buf), size=SAMPLE_DD_PG_BATCH, replace=False)]
 s = torch.stack([torch.from_numpy(b[0]) for b in batch]).float().to(device)
 u = torch.tensor([[b[1]] for b in batch], dtype=torch.float32, device=device)
 r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
 sp = torch.stack([torch.from_numpy(b[3]) for b in batch]).float().to(device)
 d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

 with torch.no_grad():
 up = agent.actor_t(sp)
 qp = agent.critic_t(torch.cat([sp, up], dim=1)).squeeze(-1)
 y = r + gamma * (1 - d) * qp

 q = agent.critic(torch.cat([s, u], dim=1)).squeeze(-1)
 loss_c = nn.functional.mse_loss(q, y)
 opt_c.zero_grad()
 loss_c.backward()
 opt_c.step()

 u_pred = agent.actor(s)
 q_a = agent.critic(torch.cat([s, u_pred], dim=1)).squeeze(-1)
 loss_a = -q_a.mean()
 opt_a.zero_grad()
 loss_a.backward()
 opt_a.step()

 agent.soft_update(0.01)

 return agent


def ddpg_dcr_scores(
 agent: DDPGDiscrete,
 env_factory: Callable[[], m.WastewaterMDP],
 episodes: int,
 device: torch.device,
) -> np.ndarray:
 agent.eval()
 out = []
 rng_m = np.random.default_rng(99)
 for _ in range(episodes):
 env = env_factory()
 rng = np.random.default_rng(int(rng_m.integers(0, 2**31)))
 obs = env.reset(rng)
 phs = [env.ph]
 for _t in range(m.T_MAX):
 with torch.no_grad():
 u = agent.actor(torch.from_numpy(obs).float().unsqueeze(0).to(device))
 a = int(DDPGDiscrete.u_to_action(u).item())
 obs, _, done, _ = env.step(a, rng)
 phs.append(env.ph)
 if done:
 break
 ph_arr = np.asarray(phs)
 out.append(float(np.mean((ph_arr >= m.PH_LO) & (ph_arr <= m.PH_HI)) * 100.0))
 return np.asarray(out, dtype=np.float64)


def markov_check_residual_autocorr(residuals: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
 """PACF proxy: normalised lag-k autocorrelation of LSTM residuals (Sec 3 Markov discussion)."""
 r = np.asarray(residuals, dtype=np.float64).ravel()
 r = r - r.mean()
 if len(r) < max_lag + 10:
 return {"note": "too_few_points"}
 var = float(np.var(r) + 1e-12)
 ac = {}
 for k in range(1, max_lag + 1):
 ac[f"acf_lag{k}"] = float(np.mean(r[k:] * r[:-k]) / var)
 return ac


def methodology_first_pass_small(
 demo_mode: bool = True,
 device: Optional[torch.device] = None,
) -> Dict[str, Any]:
 """
 One command: train short pipeline + Sec 2.3.2 + Table 6 + tiers + baselines + stats.
 """
 device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
 out = m.run_full_pipeline(demo_mode=demo_mode, device=device)

 lstm = out["lstm"]
 dph_scaler = out["dph_scaler"]
 st = out["preprocess_stats"]
 mm_cols = out["mm_cols"]
 A_T, C_T = out["A_T"], out["C_T"]
 sigma_ph, sigma_cond = out["sigma_ph"], out["sigma_cond"]
 sigma_model = out["sigma_model"]
 unc_p95 = out["unc_p95"]
 va_f = out["va_f"]

 va_ds = m.SeqDS(va_f, mm_cols, m.L_SEQ)
 lstm_metrics = lstm_val_rmse_and_ece(lstm, va_ds, dph_scaler, device)
 ds5_hz = out["ds5_hz"]
 sec232 = validate_simulator_sec232(ds5_hz, A_T, C_T, n_steps=SAMPLE_SEC232_STEPS)
 gates = table6_gates(
 lstm_metrics["surrogate_val_RMSE_pH"],
 lstm_metrics["surrogate_MAE_dph_residual"],
 sec232["median_abs_dph_err"],
 )

 def make_env(tier: str, eval_mode: bool) -> m.WastewaterMDP:
 mult = 1.0 if tier == "T1" else 1.45 if tier == "T2" else 1.15
 return m.WastewaterMDP(
 lstm,
 A_T,
 C_T,
 device,
 physics_warm=0 if eval_mode else (3_000 if demo_mode else 100_000),
 curriculum_steps=0 if eval_mode else (3_000 if demo_mode else 5_000),
 unc_p95=unc_p95,
 mc_T=8 if demo_mode else 20,
 mm=st["mm"],
 mm_cols=mm_cols,
 dph_scaler=dph_scaler,
 sigma_ph=sigma_ph * mult,
 sigma_model=sigma_model,
 sigma_cond=sigma_cond * mult,
 augment_observations=False if eval_mode else True,
 )

 n_ep = SAMPLE_TIER_EPISODES
 policy = out["policy"]

 def ef_t1():
 return make_env("T1", True)

 ppo_t1 = np.asarray(
 [m.rollout_policy(policy, ef_t1(), np.random.default_rng(i), 1, device)["DCR"] for i in range(n_ep)],
 dtype=np.float64,
 )
 rbt_t1 = _fix_rollout_dcr(ef_t1, "rbt", n_ep)
 pid_t1 = _fix_rollout_dcr(ef_t1, "pid", n_ep)
 lut_t1 = _fix_rollout_dcr(ef_t1, "lut", n_ep)

 ddpg_agent = train_ddpg_small(lambda: make_env("T1", False), device, steps=SAMPLE_DD_PG_STEPS if demo_mode else 50_000)
 ddpg_t1 = ddpg_dcr_scores(ddpg_agent, ef_t1, n_ep, device)

 tier2_ppo = np.asarray(
 [m.rollout_policy(policy, make_env("T2", True), np.random.default_rng(i), 1, device)["DCR"] for i in range(n_ep)],
 dtype=np.float64,
 )
 tier3_note = (
 "Tier-3 sim-to-real: reuse Sec 2.3.2 MAPE on DS-5 windows; full protocol uses operator-labelled trajectories."
 )

 stats4 = wilcoxon_bonferroni_four(ppo_t1, rbt_t1, pid_t1, lut_t1, ddpg_t1)

 # Residuals for Markov check (reuse val loader)
 resids = []
 lstm.eval()
 with torch.no_grad():
 for i in range(min(500, len(va_ds))):
 xa, yb, _ = va_ds[i]
 pred_s = lstm(xa.unsqueeze(0).to(device), dropout_active=False).item()
 dph_p = dph_scaler.inverse_transform(np.array([[pred_s]]))[0, 0]
 dph_t = yb.item()
 resids.append(dph_p - dph_t)
 markov = markov_check_residual_autocorr(np.asarray(resids))

 merged: Dict[str, Any] = dict(out)
 merged.update(
 {
 "sec232": sec232,
 "lstm_metrics": lstm_metrics,
 "table6_gates": gates,
 "tier1": {"ppo_DCR": ppo_t1, "rbt_DCR": rbt_t1, "pid_DCR": pid_t1, "lut_DCR": lut_t1, "ddpg_DCR": ddpg_t1},
 "tier2_ppo_DCR": tier2_ppo,
 "tier3": {"note": tier3_note, "sec232_median_abs_dph": sec232["median_abs_dph_err"]},
 "wilcoxon_bonferroni_cohen": stats4,
 "markov_residual_acf": markov,
 "ddpg_agent": ddpg_agent,
 }
 )
 return merged


if __name__ == "__main__":
 rep = methodology_first_pass_small(demo_mode=True)
 print("Sec 2.3.2:", rep["sec232"])
 print("LSTM val:", rep["lstm_metrics"])
 print("Table 6 gates:", rep["table6_gates"])
 print("Wilcoxon (first test):", rep["wilcoxon_bonferroni_cohen"]["tests"]["PPO_vs_RBT"])
