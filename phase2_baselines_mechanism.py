"""
Phase 2 utilities for simulation-only Route A:
- baseline policy mappings
- short sanity evaluations
- rollout metric instrumentation
No full PPO training is run here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import water_methodology_impl as m


def rule_based_threshold_action(ph: float, ph_lo: float = m.PH_LO, ph_hi: float = m.PH_HI) -> int:
    """Fixed threshold policy: acid if above band, alkaline if below, null inside."""
    if ph < ph_lo:
        return 8
    if ph > ph_hi:
        return 3
    return 0


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDDiscreteController:
    """
    PID mapped to 11 discrete actions.
    Gains are fixed from a documented Ziegler–Nichols closed-loop recipe.
    """

    def __init__(self, gains: PIDGains, target: float = m.PH_MID):
        self.gains = gains
        self.target = target
        self.i = 0.0
        self.prev_e = 0.0

    def reset(self) -> None:
        self.i = 0.0
        self.prev_e = 0.0

    def action(self, ph: float) -> int:
        e = self.target - ph
        self.i = float(np.clip(self.i + e, -10.0, 10.0))
        d = e - self.prev_e
        self.prev_e = e
        u = self.gains.kp * e + self.gains.ki * self.i + self.gains.kd * d
        if abs(u) < 0.05:
            return 0
        if u > 0:
            lv = int(np.clip(round(abs(u) * 2.0), 1, 5))
            return 5 + lv
        lv = int(np.clip(round(abs(u) * 2.0), 1, 5))
        return lv


class PIDDiscreteControllerLegacyNoDeadband(PIDDiscreteController):
    """
    Legacy no-deadband PID retained for traceability.
    """

    pass


class PIDDiscreteControllerDeadband(PIDDiscreteController):
    """
    PID with in-band deadband and anti-windup hold:
    - if pH within [deadband_lo, deadband_hi], return null action 0
    - freeze integrator update while idle in deadband
    """

    def __init__(
        self,
        gains: PIDGains,
        target: float = m.PH_MID,
        deadband_lo: float = 6.7,
        deadband_hi: float = 8.3,
    ):
        super().__init__(gains=gains, target=target)
        self.deadband_lo = deadband_lo
        self.deadband_hi = deadband_hi

    def action(self, ph: float) -> int:
        if self.deadband_lo <= ph <= self.deadband_hi:
            # Anti-windup hold while idle: keep integral state unchanged.
            self.prev_e = self.target - ph
            return 0

        e = self.target - ph
        self.i = float(np.clip(self.i + e, -10.0, 10.0))
        d = e - self.prev_e
        self.prev_e = e
        u = self.gains.kp * e + self.gains.ki * self.i + self.gains.kd * d
        if abs(u) < 0.05:
            return 0
        if u > 0:
            lv = int(np.clip(round(abs(u) * 2.0), 1, 5))
            return 5 + lv
        lv = int(np.clip(round(abs(u) * 2.0), 1, 5))
        return lv


def ziegler_nichols_pid_gains() -> PIDGains:
    """
    Ziegler–Nichols closed-loop tuning constants from prior Ku/Pu characterization
    on this simulator family: Ku=4.0, Pu=2.5.
    PID form:
      Kp = 0.6 Ku
      Ki = 2*Kp/Pu
      Kd = Kp*Pu/8
    """
    ku = 4.0
    pu = 2.5
    kp = 0.6 * ku
    ki = 2.0 * kp / pu
    kd = kp * pu / 8.0
    return PIDGains(kp=kp, ki=ki, kd=kd)


def build_static_lookup_table() -> Dict[str, int]:
    """
    Static compliance-band controller table.
    Philosophy: do not dose compliant water; dose only when outside the band.
    """
    return {
        "below_6.0": 10,
        "6.0_to_6.5": 8,
        "6.5_to_8.5": 0,
        "8.5_to_9.0": 3,
        "above_9.0": 5,
    }


def build_static_lookup_table_legacy_setpoint() -> Dict[str, int]:
    """
    Legacy setpoint-centric LUT retained for traceability (disabled for Route A baseline).
    """
    return {
        "e<=-1.0": 10,
        "-1.0<e<=-0.6": 9,
        "-0.6<e<=-0.3": 8,
        "-0.3<e<=-0.1": 7,
        "-0.1<e<0.1": 0,
        "0.1<=e<0.3": 2,
        "0.3<=e<0.6": 3,
        "0.6<=e<1.0": 4,
        "e>=1.0": 5,
    }


def lookup_table_action(ph: float, table: Dict[str, int], target: float = m.PH_MID) -> int:
    if ph < 6.0:
        return table["below_6.0"]
    if ph < m.PH_LO:
        return table["6.0_to_6.5"]
    if ph <= m.PH_HI:
        return table["6.5_to_8.5"]
    if ph <= 9.0:
        return table["8.5_to_9.0"]
    return table["above_9.0"]


def lookup_table_action_legacy_setpoint(ph: float, table: Dict[str, int], target: float = m.PH_MID) -> int:
    """
    Legacy setpoint-centric mapping retained for traceability (disabled for Route A baseline).
    """
    e = target - ph
    if e <= -1.0:
        return table["e<=-1.0"]
    if e <= -0.6:
        return table["-1.0<e<=-0.6"]
    if e <= -0.3:
        return table["-0.6<e<=-0.3"]
    if e <= -0.1:
        return table["-0.3<e<=-0.1"]
    if e < 0.1:
        return table["-0.1<e<0.1"]
    if e < 0.3:
        return table["0.1<=e<0.3"]
    if e < 0.6:
        return table["0.3<=e<0.6"]
    if e < 1.0:
        return table["0.6<=e<1.0"]
    return table["e>=1.0"]


class DDPGDiscrete(nn.Module):
    """Continuous actor projected to 11 discrete actions."""

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

    @staticmethod
    def u_to_action(u: torch.Tensor) -> torch.Tensor:
        return ((u + 1.0) * 0.5 * 10.0).round().clamp(0, 10).long().squeeze(-1)


def make_sim_env(
    *,
    sigma_process: float = m.SIGMA_PH,
    sigma_obs: float = m.SIGMA_PH_OBS,
    enable_curriculum_masking: bool = True,
    enable_escalation_penalty: bool = True,
    A_T: float = 1.5,
    C_T: float = 1.0,
) -> m.WastewaterMDP:
    class IdentityMM:
        def __init__(self, n: int):
            self.data_min_ = np.zeros(n, dtype=np.float64)
            self.data_max_ = np.ones(n, dtype=np.float64)

        def transform(self, x):
            return np.asarray(x, dtype=np.float64)

    class IdentityScaler:
        def inverse_transform(self, x):
            return np.asarray(x, dtype=np.float64)

    mm_cols = [
        "pH_raw",
        "dPH",
        "d1",
        "d2",
        "ph_rm1h",
        "ph_rs1h",
        "cond",
        "DO",
        "turb",
        "tsin",
        "tcos",
        "comp",
        "ph_rm6h",
        "ph_rs6h",
        "cond_rm1h",
        "cond_rs1h",
        "cond_rm6h",
        "cond_rs6h",
        "ph_mn1h",
        "ph_mx1h",
        "ph_mn6h",
        "ph_mx6h",
        "cond_mn1h",
        "cond_mx1h",
        "cond_mn6h",
        "cond_mx6h",
    ]

    return m.WastewaterMDP(
        lstm=None,
        A_T=A_T,
        C_T=C_T,
        device=torch.device("cpu"),
        physics_warm=0,
        curriculum_steps=5000,
        unc_p95=None,
        mc_T=1,
        mm=IdentityMM(len(mm_cols)),
        mm_cols=mm_cols,
        dph_scaler=IdentityScaler(),
        sigma_ph=sigma_process,
        sigma_model=0.0,
        sigma_cond=1.5,
        augment_observations=False,
        sigma_ph_obs=sigma_obs,
        enable_curriculum_masking=enable_curriculum_masking,
        enable_escalation_penalty=enable_escalation_penalty,
    )


def _rollout_episode(
    env: m.WastewaterMDP,
    rng: np.random.Generator,
    action_fn: Callable[[m.WastewaterMDP, np.random.Generator], int],
) -> Dict[str, object]:
    obs = env.reset(rng)
    phs = [env.ph]
    acts: List[int] = []
    doses: List[float] = []
    in_comp = [float(m.PH_LO <= env.ph <= m.PH_HI)]
    first_stab_step: Optional[int] = None
    for t in range(m.T_MAX):
        a = int(action_fn(env, rng))
        obs, r, done, info = env.step(a, rng)
        acts.append(a)
        doses.append(float(m.ACTION_VOLUMES_ML[a]))
        phs.append(env.ph)
        in_comp.append(float(m.PH_LO <= env.ph <= m.PH_HI))
        if first_stab_step is None and len(in_comp) >= 8 and all(v > 0.5 for v in in_comp[-8:]):
            first_stab_step = t + 1
        if done:
            break
    return {
        "phs": np.asarray(phs, dtype=np.float64),
        "acts": np.asarray(acts, dtype=np.int64),
        "doses": np.asarray(doses, dtype=np.float64),
        "first_stab_step": first_stab_step,
        "start_compliant": bool(in_comp[0] > 0.5),
    }


def rollout_metrics(ep: Dict[str, object]) -> Dict[str, float]:
    ph = ep["phs"]
    acts = ep["acts"]
    doses = ep["doses"]
    dcr = float(np.mean((ph >= m.PH_LO) & (ph <= m.PH_HI)) * 100.0)
    mpd = float(np.mean(np.abs(ph - m.PH_MID)))
    tcu = float(np.sum(doses))
    # Bounded CER definition (Phase 2 fix 2):
    # compliance efficiency per unit reagent with a +1 denominator offset to keep
    # zero-dose episodes finite and comparable.
    cer = float(dcr / (1.0 + tcu))
    d1 = np.diff(ph)
    oec = float(np.sum((np.sign(d1[1:]) != np.sign(d1[:-1])) & (np.abs(d1[1:]) > 0.2)))
    pst = float(ep["first_stab_step"] if ep["first_stab_step"] is not None else m.T_MAX)
    pdcr = float(np.mean((acts <= 7).astype(np.float64)) if len(acts) else 0.0)
    return {
        "DCR": dcr,
        "MPD": mpd,
        "TCU": tcu,
        "CER": cer,
        "OEC": oec,
        "PST": pst,
        "PDCR": pdcr,
        "STG": float("nan"),  # not applicable / future work in Route A
    }


def evaluate_policy_short(
    action_fn: Callable[[m.WastewaterMDP, np.random.Generator], int],
    episodes: int = 40,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = make_sim_env()
    dcrs = []
    starts = []
    for _ in range(episodes):
        ep = _rollout_episode(env, rng, action_fn)
        dcrs.append(float(np.mean((ep["phs"] >= m.PH_LO) & (ep["phs"] <= m.PH_HI)) * 100.0))
        starts.append(1.0 if ep["start_compliant"] else 0.0)
    return {
        "episodes": float(episodes),
        "dcr_mean": float(np.mean(dcrs)),
        "dcr_sd": float(np.std(dcrs, ddof=1)),
        "start_compliant_frac": float(np.mean(starts)),
    }


def policy_random(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
    return int(rng.integers(0, 11))


def policy_null(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
    return 0


def policy_rule_based(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
    return rule_based_threshold_action(env.ph)


def make_policy_pid(gains: PIDGains) -> Callable[[m.WastewaterMDP, np.random.Generator], int]:
    ctrl = PIDDiscreteControllerDeadband(gains)

    def _act(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
        return ctrl.action(env.ph)

    _act.reset = ctrl.reset  # type: ignore[attr-defined]
    return _act


def make_policy_pid_legacy_no_deadband(gains: PIDGains) -> Callable[[m.WastewaterMDP, np.random.Generator], int]:
    ctrl = PIDDiscreteControllerLegacyNoDeadband(gains)

    def _act(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
        return ctrl.action(env.ph)

    _act.reset = ctrl.reset  # type: ignore[attr-defined]
    return _act


def make_policy_lut(table: Dict[str, int]) -> Callable[[m.WastewaterMDP, np.random.Generator], int]:
    def _act(env: m.WastewaterMDP, rng: np.random.Generator) -> int:
        return lookup_table_action(env.ph, table)

    return _act


def train_ddpg_probe(steps: int = 1000, seed: int = 7) -> Dict[str, float]:
    """
    Minimal run-only check that DDPG wiring executes in current environment.
    This is not Phase-3 training.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_sim_env()
    agent = DDPGDiscrete()
    opt_a = optim.Adam(agent.actor.parameters(), lr=1e-4)
    opt_c = optim.Adam(agent.critic.parameters(), lr=1e-3)
    buf: List[tuple] = []
    obs = env.reset(rng).astype(np.float32)
    losses = []
    for _ in range(steps):
        with torch.no_grad():
            u = agent.actor(torch.from_numpy(obs).float().unsqueeze(0))
            u = torch.clamp(u + 0.1 * torch.randn_like(u), -1, 1)
            a = int(DDPGDiscrete.u_to_action(u).item())
        obs2, r, done, _ = env.step(a, rng)
        buf.append((obs.copy(), float(u.item()), float(r), obs2.copy(), float(done)))
        if len(buf) > 4096:
            buf.pop(0)
        obs = env.reset(rng).astype(np.float32) if done else obs2.astype(np.float32)
        if len(buf) < 64:
            continue
        idx = rng.choice(len(buf), size=64, replace=False)
        batch = [buf[i] for i in idx]
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        u_b = torch.tensor([[b[1]] for b in batch], dtype=torch.float32)
        r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        sp = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        with torch.no_grad():
            up = agent.actor(sp)
            qn = agent.critic(torch.cat([sp, up], dim=1)).squeeze(-1)
            y = r_b + m.GAMMA * (1.0 - d_b) * qn
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
        losses.append(float(loss_c.item()))
    return {
        "steps": float(steps),
        "buffer_size": float(len(buf)),
        "critic_loss_mean": float(np.mean(losses)) if losses else float("nan"),
    }

