"""
Implementation of *Methodology v.01* - MBRL wastewater dosing pipeline.

Section mapping
---------------
- Table 1 stages 1-2: `synth_*`, `reconstruct_actions`, `preprocess_monitor`,
 `f_titration`, `LSTMSurrogate`, `train_lstm`.
- Table 1 stages 3-4: `WastewaterMDP` (state, Table 9 actions, Table 10 reward,
 Eq. 5 hybrid transition), `ppo_train` (Table 12 PPO), curriculum masking
 (Sec 5.3), MC-Dropout gate (Table 13), running reward norm (Table 13),
 hard pH termination (Table 13).
- Table 1 stage 5-6: `rollout_policy`, `rollout_controller`, `wilcoxon_report`;
 DS-5 resampling helper `ds5_downsample_15m` for Tier-3 style inputs.

Compliance vs *Methodology v.01* (audit)
----------------------------------------
**Implemented to match the text:** Newton-Raphson alkalinity solve to |ΔpH|<1e⁻⁶
(Sec 2.3.1); A_T / C_T from DS‑2 + DS‑1 row with Henry-style temperature factor;
Table 4 P1-P6 (interpolation, bounded ff/bf, >20% missing drop, P2 3σ/24h + dual
physical removal, `unc_stat` flag, P3 MinMax on inputs + **StandardScaler on
ΔpH for LSTM only**, P5 **mean/std/min/max** at 1h+6h for pH and conductivity
(sixteen rolling features beyond the base monitoring block; Table 8 RL obs still
uses the same 13-D subset as before), P6 chronological split); Table 8 state in
RL from the **same** MinMax transform as monitoring; Eq. 5 σ_aleatoric from
**DS‑5 diffs** + σ_model from **LSTM val residuals**; Table 13 **observation
noise** on normalised pH & cond during PPO training; **Dyna** refresh every 500k
steps (hook in `ppo_train`); curriculum masking, MC‑Dropout gate, reward running
norm, hybrid physics→LSTM; PPO Table 12; RBT & PID Table 15; Wilcoxon; eval env
`physics_warm=0` for surrogate-only rollouts.

**Bundled elsewhere:** `water_experiments_small.py` runs a **short-sample** first
pass (Sec 2.3.2 diagnostic, Table 6 gates, Table 16 extras, Tier‑2/3 smoke, DDPG, LUT,
Wilcoxon + Bonferroni + Cohen’s d). Extended ablations (Table 17), multi-seed runs at
full PPO budget, and external MLOps are **out of scope** for this repository; see
`Implementation_and_Remaining_Work.md` for manuscript vs code scope.

**Table 2 public data:** `water_rdi_loaders.build_table2_mixed` pulls **USGS NWIS**
IV/DV when the network is available (DS-1 / DS-4 / DS-5), uses the bundled WQP-derived
**DS-2** CSV when present (see `data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv`), and
optional ``WATER_DS3_CSV``; otherwise bundled WQP effluent proxy DS-3 when present.
``run_full_pipeline`` calls that builder unless ``WATER_USE_SYNTH_ONLY=1``.
Set ``WATER_TABLE2_REQUIRE_REAL=1`` to abort if any role still falls back to synthetic.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Table / section constants (methodology)
# ---------------------------------------------------------------------------

CUSUM_K = 0.05
CUSUM_H = 0.30
CUSUM_MERGE_MINUTES = 15.0
MIN_EVENT_DELTA_PH = 0.1
INVERSE_ASSIGNMENT_MAX_RESIDUAL = 0.5

ACTION_VOLUMES_ML = np.array([0, 5, 12, 30, 75, 180, 5, 12, 30, 75, 180], dtype=np.float64)
DOSE_MAX_ML = float(ACTION_VOLUMES_ML.max())

PH_LO, PH_HI = 6.5, 8.5
PH_MID = 7.5
PH_HARD_LO, PH_HARD_HI = 2.0, 12.0

# Table 6 - surrogate acceptance gates (*Methodology v.01* manuscript)
TABLE6_GATE_RMSE = 0.10 # surrogate vs simulator / val, pH units
TABLE6_GATE_MAE_DPH = 0.07 # MAE on ΔpH residuals
TABLE6_GATE_SEC232_MEDIAN_DPH = 0.25 # Sec 2.3.2 sim-vs-obs median |ΔpH|, pH units

W_COMP, W_DEV, W_DOSE, W_OVER, W_ESC = 2.0, -1.0, -0.3, -0.5, -0.1
W_UNC = -0.5

GAMMA = 0.99
GAE_LAMBDA = 0.95
T_MAX = 480

V_SYSTEM_L = 1000.0
C_REAGENT_M = 0.1

PK1, PK2, PKW = 6.35, 10.33, 14.0

# Sensor noise (Table 13) - fitted from DS-5 when available; defaults match text
SIGMA_PH = 0.02
SIGMA_COND = 1.5 # μS/cm

# Curriculum masking denominator (Sec 5.3)
PH_RANGE_MASK = 7.5

# LSTM training defaults (Table 5) - override lengths via run_full_pipeline for demo
LSTM_LR = 1e-3
LSTM_BATCH = 64
LSTM_EPOCHS = 200
LSTM_PATIENCE = 15
HUBER_DELTA = 1.0
L_SEQ = 48

# PPO (Table 12)
ENT_START, ENT_END = 0.01, 0.001
ENT_DECAY_STEPS = 4_000_000
PPO_CLIP = 0.2
PPO_EPOCHS = 4
MAX_GRAD_NORM = 0.5

# Table 2 - RDI dataset inventory (*Methodology v.01* wording)
RDI_TABLE2: Dict[str, Dict[str, str]] = {
 "DS-1": {
 "name": "Large-Scale WQ Monitoring (National)",
 "key_parameters": "pH, turbidity, DO, conductivity, temperature, nutrients",
 "resolution": "15-60 min; multi-year; national rivers & industrial outfalls",
 "role": "Primary surrogate training data; dosing event reconstruction source",
 },
 "DS-2": {
 "name": "Regional Industrial Surface & Groundwater",
 "key_parameters": "pH, TDS, BOD, COD, hardness, heavy metals",
 "resolution": "Daily to weekly; industrially affected catchments",
 "role": "Regulatory constraint calibration; buffering capacity parameter estimation",
 },
 "DS-3": {
 "name": "Survey-Based Industrial Effluent Reports",
 "key_parameters": "pH, discharge class, effluent type, compliance status",
 "resolution": "Cross-sectional; structured regulatory records",
 "role": "Discharge standard threshold validation; compliance label ground-truth",
 },
 "DS-4": {
 "name": "Global Multi-Decadal WQ (GEMS/Water-UN)",
 "key_parameters": "pH, DO, nitrate, phosphate, conductivity (1970-2023)",
 "resolution": "Monthly; 150+ countries; surface water",
 "role": "Out-of-distribution (OOD) generalization test set, never seen in training",
 },
 "DS-5": {
 "name": "IoT Sensor-Based WQ (1 Hz real-time)",
 "key_parameters": "pH, temperature, turbidity; calibrated sensor with drift logs",
 "resolution": "~1 min-1 Hz continuous (e.g. KU-MWQ or NWIS IV); sensor noise & drift characterization",
 "role": "(a) Noise model fitting; (b) Sim-to-real validation, held-out real trajectories",
 },
}


def rdi_dataset_name(code: str) -> str:
 """Return the Table 2 *Name / Source* string for ``DS-1`` … ``DS-5``."""
 return RDI_TABLE2.get(code, {}).get("name", code)


def rdi_table2_lines() -> List[str]:
 """One human-readable line per dataset (for logging or notebooks)."""
 lines: List[str] = []
 for code, meta in RDI_TABLE2.items():
 lines.append(f"{code} - {meta['name']}: {meta['role']}")
 return lines


def estimate_AT_CT_from_ds2(
 ds2: pd.DataFrame,
 ds1_row: Union[pd.Series, Mapping[str, object], float, int, np.floating, np.integer],
) -> Tuple[float, float]:
 """Sec 2.3.1 - A_T from **DS-2** (Regional Industrial Surface & Groundwater) hardness + conductivity;
 C_T from **DS-1** (Large-Scale WQ Monitoring (National)) + Henry.

 ``ds1_row`` may be a **DS-1** row (Series or dict-like) or a scalar conductivity (µS/cm), e.g. a
 column median, in which case temperature defaults to 25 °C.
 """
 hard = float(ds2["hardness_mgL"].median())
 if isinstance(ds1_row, pd.Series) or isinstance(ds1_row, Mapping):
 cond = float(ds1_row.get("conductivity_uScm", ds2["conductivity_uScm"].median())) # type: ignore[union-attr]
 T_C = float(ds1_row.get("temperature_C", 25.0)) # type: ignore[union-attr]
 else:
 cond = float(ds1_row)
 T_C = 25.0
 # Alkalinity (meq/L): carbonate hardness contribution + ionic strength proxy (textbook-style)
 A_T = 1.8 * (hard / 100.0) + 2.2 * (cond / 1000.0)
 A_T = float(np.clip(A_T, 0.5, 12.0))
 T_K = 273.15 + T_C
 # Henry's law for CO2(aq) ~ kH(T) * pCO2; use conductivity as CO2-carbonate coupling proxy (methodology narrative)
 kH = 0.034 * np.exp(2400.0 * (1.0 / 298.15 - 1.0 / T_K))
 C_T = 0.018 * (cond / 1000.0) + 11.0 * kH
 C_T = float(np.clip(C_T, 0.5, 25.0))
 return A_T, C_T


def alkalinity_from_ph(pH: float, C_T_mM: float) -> float:
 H = 10.0 ** (-pH)
 OH = 10.0 ** (pH - PKW)
 K1, K2 = 10.0 ** (-PK1), 10.0 ** (-PK2)
 denom = H * H + K1 * H + K1 * K2
 alpha1 = (K1 * H) / denom
 alpha2 = (K1 * K2) / denom
 HCO3 = C_T_mM * alpha1
 CO3 = C_T_mM * alpha2
 return HCO3 + 2 * CO3 + OH - H


def solve_ph_newton_raphson(
 A_T_meq_L: float,
 C_T_mM: float,
 pH0: float = 7.0,
 tol: float = 1e-6,
 max_iter: int = 100,
) -> float:
 """Sec 2.3.1 - Newton-Raphson on alkalinity closure until |ΔpH| < tol."""
 ph = float(np.clip(pH0, 2.0, 12.0))
 for _ in range(max_iter):
 f = alkalinity_from_ph(ph, C_T_mM) - A_T_meq_L
 dfdh = (alkalinity_from_ph(ph + 1e-5, C_T_mM) - alkalinity_from_ph(ph - 1e-5, C_T_mM)) / 2e-5
 if abs(dfdh) < 1e-14:
 break
 step = f / dfdh
 ph_new = ph - step
 if abs(ph_new - ph) < tol:
 return float(np.clip(ph_new, 0.0, 14.0))
 ph = float(np.clip(ph_new, 2.0, 12.0))
 return float(np.clip(ph, 0.0, 14.0))


def solve_ph_from_TA_CT(A_T_meq_L: float, C_T_mM: float, pH_guess: float = 7.0) -> float:
 """Backward-compatible name; uses Newton-Raphson (methodology) instead of brentq."""
 return solve_ph_newton_raphson(A_T_meq_L, C_T_mM, pH0=pH_guess)


def moles_from_action(action: int) -> float:
 vol_ml = ACTION_VOLUMES_ML[int(action)]
 if vol_ml <= 0:
 return 0.0
 mol_r = (vol_ml / 1000.0) * C_REAGENT_M
 if 1 <= action <= 5:
 return -2.0 * mol_r
 return mol_r


def f_titration(pH: float, action: int, A_T: float, C_T: float, V_L: float = V_SYSTEM_L) -> float:
 """Eq. 2 - discrete pH after dose; Newton-Raphson with |ΔpH| < 1e⁻⁶ (Sec 2.3.1)."""
 if int(action) == 0:
 return float(np.clip(pH, 0.0, 14.0))
 d_mol = moles_from_action(int(action))
 A_curr = alkalinity_from_ph(pH, C_T)
 dA = (d_mol / V_L) * 1000.0
 A_new = float(np.clip(A_curr + dA, 1e-3, 500.0))
 return solve_ph_newton_raphson(A_new, C_T, pH0=pH, tol=1e-6)


# ---------------------------------------------------------------------------
# Synthetic datasets (Table 2 - RDI roles; replace with file loaders for publication)
# ---------------------------------------------------------------------------


def synth_ds1(n_rows: int, seed: int = 0) -> pd.DataFrame:
 """**DS-1** - Large-Scale WQ Monitoring (National) (Table 2); synthetic 15-min proxy."""
 rng = np.random.default_rng(seed)
 t0 = pd.Timestamp("2020-01-01", tz="UTC")
 idx = pd.date_range(t0, periods=n_rows, freq="15min")
 base = 7.2 + 0.35 * np.sin(np.linspace(0, 60, n_rows))
 walk = np.cumsum(rng.normal(0, 0.018, n_rows))
 ph = base + walk
 for _ in range(max(1, n_rows // 500)):
 j = rng.integers(20, n_rows - 8)
 ph[j : j + 4] += rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 0.9)
 ph = np.clip(ph, 4.0, 11.0)
 cond = 800 + 35 * (ph - 7.0) + rng.normal(0, 12, n_rows)
 df = pd.DataFrame(
 {
 "timestamp": idx,
 "pH": ph,
 "conductivity_uScm": cond,
 "DO_mgL": rng.uniform(3, 9, n_rows),
 "turbidity_NTU": rng.uniform(2, 28, n_rows),
 "temperature_C": rng.uniform(18.0, 28.0, n_rows),
 }
 )
 df["hour"] = df["timestamp"].dt.hour.astype(float)
 return df


synth_rdi_ds1_large_scale_wq_monitoring_national = synth_ds1


def synth_ds2(seed: int = 1) -> pd.DataFrame:
 """**DS-2** - Regional Industrial Surface & Groundwater (Table 2); synthetic proxy."""
 rng = np.random.default_rng(seed)
 n = 400
 return pd.DataFrame(
 {
 "hardness_mgL": rng.uniform(50, 300, n),
 "conductivity_uScm": rng.uniform(400, 2000, n),
 }
 )


synth_rdi_ds2_regional_industrial_surface_groundwater = synth_ds2


def synth_ds3(seed: int = 4, n: int = 600) -> pd.DataFrame:
 """**DS-3** - Survey-Based Industrial Effluent Reports (Table 2); synthetic cross-sectional proxy."""
 rng = np.random.default_rng(seed)
 return pd.DataFrame(
 {
 "pH": rng.uniform(5.5, 9.5, n),
 "discharge_class": rng.choice(np.array(["I", "II", "III"], dtype=object), n),
 "effluent_type": rng.choice(np.array(["organic", "inorganic", "mixed"], dtype=object), n),
 "compliance_status": rng.choice(np.array([0, 1], dtype=np.int64), n, p=[0.12, 0.88]),
 }
 )


synth_rdi_ds3_survey_based_industrial_effluent_reports = synth_ds3


def synth_ds5_hz(seconds: int, seed: int = 2) -> pd.DataFrame:
 """**DS-5** - IoT Sensor-Based WQ (1 Hz real-time) (Table 2); synthetic proxy."""
 rng = np.random.default_rng(seed)
 t0 = pd.Timestamp("2024-06-01", tz="UTC")
 idx = pd.date_range(t0, periods=seconds, freq="s")
 drift = np.linspace(0, float(rng.choice([-1.7, 1.7])), seconds)
 ph = np.clip(7.4 + drift + rng.normal(0, SIGMA_PH, seconds), 4.5, 10.5)
 cond = 900.0 + 0.08 * np.arange(seconds) + rng.normal(0, 2.0, seconds)
 return pd.DataFrame(
 {
 "timestamp": idx,
 "pH": ph,
 "conductivity_uScm": cond,
 "temperature_C": rng.normal(22.0, 0.4, seconds),
 "turbidity_NTU": rng.uniform(1, 8, seconds),
 }
 )


def synth_ds4_monthly(n: int = 200, seed: int = 3) -> pd.DataFrame:
 """**DS-4** - Global Multi-Decadal WQ (GEMS/Water-UN) (Table 2); synthetic monthly proxy."""
 rng = np.random.default_rng(seed)
 idx = pd.date_range("1990-01-01", periods=n, freq="MS", tz="UTC")
 return pd.DataFrame(
 {
 "timestamp": idx,
 "pH": rng.uniform(5.5, 9.0, n),
 "conductivity_uScm": rng.uniform(200, 1500, n),
 }
 )


synth_rdi_ds5_iot_sensor_wq_1hz = synth_ds5_hz
synth_rdi_ds4_global_multidecadal_wq_gems_water_un = synth_ds4_monthly

# Optional registry: call by Table 2 ID (synthetic loaders only until RDI files are wired).
SYNTH_LOADERS = {
 "DS-1": synth_ds1,
 "DS-2": synth_ds2,
 "DS-3": synth_ds3,
 "DS-4": synth_ds4_monthly,
 "DS-5": synth_ds5_hz,
}


def ds5_downsample_15m(df_hz: pd.DataFrame) -> pd.DataFrame:
 """Table 4 P4 - median aggregate per 15-minute window (**DS-5** 1 Hz → 15 min)."""
 x = df_hz.set_index("timestamp")
 r = x.resample("15min").median().dropna(how="any").reset_index()
 return r


# ---------------------------------------------------------------------------
# CUSUM + inverse assignment (Sec 2.4)
# ---------------------------------------------------------------------------


def cusum_events(ph: np.ndarray, dt_minutes: float = 15.0) -> np.ndarray:
 n = len(ph)
 d = np.diff(ph, prepend=ph[0])
 cp = np.zeros(n)
 cm = np.zeros(n)
 for t in range(1, n):
 cp[t] = max(0.0, cp[t - 1] + d[t] - CUSUM_K)
 cm[t] = max(0.0, cm[t - 1] - d[t] - CUSUM_K)
 raw = (cp > CUSUM_H) | (cm > CUSUM_H)
 merge_steps = max(1, int(round(CUSUM_MERGE_MINUTES / dt_minutes)))
 merged = np.zeros_like(raw)
 i = 0
 while i < n:
 if raw[i]:
 j = min(n - 1, i + merge_steps)
 merged[i : j + 1] = True
 i = j + 1
 else:
 i += 1
 ev = np.zeros_like(raw, dtype=bool)
 idxs = np.where(merged)[0]
 for t in idxs:
 lo, hi = max(0, t - 1), min(n - 1, t + 2)
 if abs(ph[min(hi, n - 1)] - ph[lo]) >= MIN_EVENT_DELTA_PH:
 ev[t] = True
 return ev


def _cusum_dosing_windows(ph: np.ndarray) -> List[Tuple[int, int, int]]:
 """Merged CUSUM dosing windows: (t_start, t_end, t_next) with t_next = successor index for inverse map."""
 ev = cusum_events(ph)
 n = len(ph)
 out: List[Tuple[int, int, int]] = []
 t = 0
 while t < n:
 if not ev[t]:
 t += 1
 continue
 t0 = t
 while t + 1 < n and ev[t + 1]:
 t += 1
 t_end = t
 t_next = min(n - 1, t_end + 1)
 out.append((t0, t_end, t_next))
 t = t_end + 1
 return out


def assign_action_inverse_detail(pH_t: float, pH_next: float, A_T: float, C_T: float) -> Tuple[int, float]:
 """Return chosen action and best residual (Eq. 4); action 0 means discarded (residual > threshold)."""
 best_a, best_res = 0, 1e9
 for a in range(1, 11):
 pred = f_titration(pH_t, a, A_T, C_T)
 res = abs(pred - pH_next)
 if res < best_res:
 best_a, best_res = a, res
 if best_res > INVERSE_ASSIGNMENT_MAX_RESIDUAL:
 return 0, float(best_res)
 return int(best_a), float(best_res)


def assign_action_inverse(pH_t: float, pH_next: float, A_T: float, C_T: float) -> int:
 a, _ = assign_action_inverse_detail(pH_t, pH_next, A_T, C_T)
 return int(a)


def reconstruct_actions(ds1: pd.DataFrame, A_T: float, C_T: float) -> np.ndarray:
 """**DS-1** monitoring - one label per merged dosing window (Sec 2.4.1 merge rule): label the first index only."""
 ph = ds1["pH"].to_numpy(dtype=np.float64)
 acts = np.zeros(len(ph), dtype=np.int64)
 for t0, _t_end, t_next in _cusum_dosing_windows(ph):
 acts[t0] = assign_action_inverse(float(ph[t0]), float(ph[t_next]), A_T, C_T)
 return acts


def cusum_false_positive_rate_proxy(n_steps: int = 8000, seed: int = 42) -> float:
 """Table 3 - CUSUM flag rate on flat pH + i.i.d. noise (no dosing); proxy for false positives."""
 rng = np.random.default_rng(seed)
 ph = 7.0 + rng.normal(0, 0.012, n_steps)
 return float(cusum_events(ph).mean())


def table3_reconstruction_metrics(ds1: pd.DataFrame, A_T: float, C_T: float) -> Dict[str, float]:
 """Table 3 - reconstruction quality on **DS-1** (Large-Scale WQ Monitoring (National))."""
 ph = ds1["pH"].to_numpy(dtype=np.float64)
 windows = _cusum_dosing_windows(ph)
 acts = np.zeros(len(ph), dtype=np.int64)
 discarded = 0
 inv_abs: List[float] = []
 for t0, _t_end, t_next in windows:
 a, br = assign_action_inverse_detail(float(ph[t0]), float(ph[t_next]), A_T, C_T)
 acts[t0] = a
 if a == 0:
 discarded += 1
 else:
 pred = f_titration(float(ph[t0]), a, A_T, C_T)
 inv_abs.append(abs(pred - float(ph[t_next])))
 nw = max(1, len(windows))
 return {
 "n_timesteps": float(len(ph)),
 "n_cusum_windows": float(len(windows)),
 "discard_rate": float(discarded / nw),
 "null_action_fraction": float((acts == 0).mean()),
 "inverse_residual_mae": float(np.mean(inv_abs)) if inv_abs else float("nan"),
 "cusum_flag_timestep_rate": float(cusum_events(ph).mean()),
 "cusum_fp_proxy_flat_noise": float(cusum_false_positive_rate_proxy()),
 }


# ---------------------------------------------------------------------------
# Preprocessing P1-P6 (Table 4) + chronological split
# ---------------------------------------------------------------------------


def preprocess_monitor(
 df: pd.DataFrame, stats: Optional[Dict] = None, fit: bool = True
) -> Tuple[pd.DataFrame, Dict]:
 """Table 4 P1-P6 - leakage-safe when `fit=True` on train only; val/test pass `stats=`."""
 df = df.sort_values("timestamp").reset_index(drop=True).copy()
 if "temperature_C" not in df.columns:
 df["temperature_C"] = 25.0
 # IoT-only traces (e.g. KU-MWQ) may omit conductivity / DO; use neutral placeholders before ffill.
 if "conductivity_uScm" not in df.columns:
 df["conductivity_uScm"] = 900.0
 if "DO_mgL" not in df.columns:
 df["DO_mgL"] = 8.0
 if "turbidity_NTU" not in df.columns:
 df["turbidity_NTU"] = 5.0
 cols_mon = ["pH", "conductivity_uScm", "DO_mgL", "turbidity_NTU"]
 for c in cols_mon:
 if c in df.columns:
 df[c] = df[c].interpolate(method="linear", limit=3)
 # P1: forward-fill bounded [0,14] for pH for remaining gaps (4-12 step regime proxy)
 df["pH"] = df["pH"].clip(lower=0.0, upper=14.0)
 df["pH"] = df["pH"].ffill(limit=12).bfill(limit=12)
 if df["pH"].isna().mean() > 0.2:
 df = df.dropna(subset=["pH"])
 df["conductivity_uScm"] = df["conductivity_uScm"].ffill(limit=12).bfill(limit=12)
 df["DO_mgL"] = df["DO_mgL"].ffill(limit=12).bfill(limit=12)
 df["turbidity_NTU"] = df["turbidity_NTU"].ffill(limit=12).bfill(limit=12)

 ph = df["pH"].to_numpy(dtype=np.float64)
 cond = df["conductivity_uScm"].to_numpy(dtype=np.float64)
 dox = df["DO_mgL"].to_numpy(dtype=np.float64)
 turb = df["turbidity_NTU"].to_numpy(dtype=np.float64)

 # P2: 24h rolling (96 steps @ 15 min) for statistical outlier rule
 w24 = 96
 s_ph = pd.Series(ph)
 m24 = s_ph.rolling(w24, min_periods=4).mean().to_numpy()
 sd24 = s_ph.rolling(w24, min_periods=4).std().replace(0, np.nan).fillna(1e-6).to_numpy()
 stat_z = np.abs(ph - m24) / sd24
 phys_bad = (ph < 0) | (ph > 14) | (dox < 0) | (cond < 0)
 stat_bad = stat_z > 3.0
 unc_stat = stat_bad & (~phys_bad)
 drop_both = stat_bad & phys_bad
 keep = ~drop_both
 unc_series = unc_stat[np.asarray(keep)]
 df = df.loc[keep].reset_index(drop=True)
 ph = df["pH"].to_numpy(dtype=np.float64)
 cond = df["conductivity_uScm"].to_numpy(dtype=np.float64)
 dox = df["DO_mgL"].to_numpy(dtype=np.float64)
 turb = df["turbidity_NTU"].to_numpy(dtype=np.float64)

 w1, w6 = 4, 24
 s_ph = pd.Series(ph)
 s_cd = pd.Series(cond)
 ph_rm1h = s_ph.rolling(w1, min_periods=1).mean().to_numpy()
 ph_rs1h = s_ph.rolling(w1, min_periods=1).std().fillna(0.0).to_numpy()
 ph_rm6h = s_ph.rolling(w6, min_periods=1).mean().to_numpy()
 ph_rs6h = s_ph.rolling(w6, min_periods=1).std().fillna(0.0).to_numpy()
 cond_rm1h = s_cd.rolling(w1, min_periods=1).mean().to_numpy()
 cond_rs1h = s_cd.rolling(w1, min_periods=1).std().fillna(0.0).to_numpy()
 cond_rm6h = s_cd.rolling(w6, min_periods=1).mean().to_numpy()
 cond_rs6h = s_cd.rolling(w6, min_periods=1).std().fillna(0.0).to_numpy()
 # P5 - rolling min/max at 1h and 6h (completes four statistics × two windows × two signals)
 ph_mn1h = s_ph.rolling(w1, min_periods=1).min().to_numpy()
 ph_mx1h = s_ph.rolling(w1, min_periods=1).max().to_numpy()
 ph_mn6h = s_ph.rolling(w6, min_periods=1).min().to_numpy()
 ph_mx6h = s_ph.rolling(w6, min_periods=1).max().to_numpy()
 cond_mn1h = s_cd.rolling(w1, min_periods=1).min().to_numpy()
 cond_mx1h = s_cd.rolling(w1, min_periods=1).max().to_numpy()
 cond_mn6h = s_cd.rolling(w6, min_periods=1).min().to_numpy()
 cond_mx6h = s_cd.rolling(w6, min_periods=1).max().to_numpy()

 d1 = np.diff(ph, prepend=ph[0])
 d2 = np.diff(d1, prepend=d1[0])
 hour = df["timestamp"].dt.hour.to_numpy()
 tsin = np.sin(2 * math.pi * hour / 24.0)
 tcos = np.cos(2 * math.pi * hour / 24.0)
 comp = ((ph >= PH_LO) & (ph <= PH_HI)).astype(np.float32)
 frame = pd.DataFrame(
 {
 "timestamp": df["timestamp"].values,
 "pH_phys": ph,
 "pH_raw": ph,
 "dPH": ph - PH_MID,
 "d1": d1,
 "d2": d2,
 "ph_rm1h": ph_rm1h,
 "ph_rs1h": ph_rs1h,
 "cond": cond,
 "DO": dox,
 "turb": turb,
 "tsin": tsin,
 "tcos": tcos,
 "comp": comp,
 "unc_stat": unc_series.astype(np.float32),
 "ph_rm6h": ph_rm6h,
 "ph_rs6h": ph_rs6h,
 "cond_rm1h": cond_rm1h,
 "cond_rs1h": cond_rs1h,
 "cond_rm6h": cond_rm6h,
 "cond_rs6h": cond_rs6h,
 "ph_mn1h": ph_mn1h,
 "ph_mx1h": ph_mx1h,
 "ph_mn6h": ph_mn6h,
 "ph_mx6h": ph_mx6h,
 "cond_mn1h": cond_mn1h,
 "cond_mx1h": cond_mx1h,
 "cond_mn6h": cond_mn6h,
 "cond_mx6h": cond_mx6h,
 }
 )
 # Table 4 P5: mean/std/min/max at 1h and 6h for pH and conductivity (16 roll features; Table 8 uses 13-D subset)
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
 # Long gaps in raw turbidity / conductivity can survive interpolate+ffill; rolling stats must be finite
 # for MinMax + LSTM (NaN inputs → NaN loss → corrupted weights).
 vals = frame[mm_cols].replace([np.inf, -np.inf], np.nan)
 for c in mm_cols:
 vals[c] = vals[c].ffill().bfill()
 col_med = vals.median(numeric_only=True)
 vals = vals.fillna(col_med).fillna(0.0)
 frame[mm_cols] = vals
 if fit:
 mm = MinMaxScaler()
 mm.fit(frame[mm_cols].values)
 stats = {"mm": mm, "mm_cols": mm_cols}
 mm = stats["mm"]
 mm_cols = stats["mm_cols"]
 frame[mm_cols] = mm.transform(frame[mm_cols].values)
 return frame, stats


def chronological_split(n: int) -> Tuple[slice, slice, slice]:
 i1 = int(0.70 * n)
 i2 = int(0.85 * n)
 return slice(0, i1), slice(i1, i2), slice(i2, n)


def attach_actions(frame: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
 """Add integer `a` and scaled `a_prev` (Table 8) aligned to frame rows."""
 fr = frame.copy()
 a = np.asarray(actions, dtype=np.int64)
 ap = np.roll(a, 1)
 ap[0] = 0
 fr["a"] = a[: len(fr)]
 fr["a_prev"] = (ap[: len(fr)] / 10.0).astype(np.float64)
 return fr


TABLE8_COLS = [
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
 "a_prev",
]
MM_LEN = 26 # len(mm_cols) in preprocess_monitor (P5 includes min/max rollups)
LSTM_IN_DIM = MM_LEN + 2 # + a_prev + a_t (Table 5)


def estimate_sigmas_from_ds5(ds5_hz: pd.DataFrame) -> Tuple[float, float]:
 """Table 13 - σ_pH and σ_cond from **DS-5** (IoT Sensor-Based WQ, 1 Hz) high-frequency residual spread."""
 dp = ds5_hz["pH"].diff().dropna().to_numpy()
 sp = float(np.std(dp)) if len(dp) > 2 else SIGMA_PH
 sp = float(np.clip(sp, 0.005, 0.5))
 if "conductivity_uScm" in ds5_hz.columns:
 dc = ds5_hz["conductivity_uScm"].diff().dropna().to_numpy()
 sc = float(np.std(dc)) if len(dc) > 2 else SIGMA_COND
 else:
 sc = SIGMA_COND
 sc = float(np.clip(sc, 0.05, 200.0))
 return sp, sc


def _roll_mean(x: np.ndarray, w: int) -> float:
 w = min(w, len(x))
 return float(np.mean(x[-w:])) if w > 0 else float(x[-1])


def _roll_std(x: np.ndarray, w: int) -> float:
 w = min(w, len(x))
 if w < 2:
 return 0.0
 return float(np.std(x[-w:]))


def _roll_min(x: np.ndarray, w: int) -> float:
 w = min(w, len(x))
 return float(np.min(x[-w:])) if w > 0 else float(x[-1])


def _roll_max(x: np.ndarray, w: int) -> float:
 w = min(w, len(x))
 return float(np.max(x[-w:])) if w > 0 else float(x[-1])


def mm_row_from_histories(
 ph_hist: List[float],
 cond_hist: List[float],
 do_hist: List[float],
 turb_hist: List[float],
 hour: float,
 mm: MinMaxScaler,
 mm_cols: List[str],
) -> np.ndarray:
 """Build one MinMax-normalised row matching `preprocess_monitor` (train-fitted `mm`)."""
 ph = np.asarray(ph_hist, dtype=np.float64)
 cd = np.asarray(cond_hist, dtype=np.float64)
 dox = np.asarray(do_hist, dtype=np.float64)
 tb = np.asarray(turb_hist, dtype=np.float64)
 p = float(ph[-1])
 vals = {
 "pH_raw": p,
 "dPH": p - PH_MID,
 "d1": float(ph[-1] - ph[-2]) if len(ph) > 1 else 0.0,
 "d2": 0.0,
 "ph_rm1h": _roll_mean(ph, 4),
 "ph_rs1h": _roll_std(ph, 4),
 "cond": float(cd[-1]),
 "DO": float(dox[-1]),
 "turb": float(tb[-1]),
 "tsin": math.sin(2 * math.pi * hour / 24.0),
 "tcos": math.cos(2 * math.pi * hour / 24.0),
 "comp": float(PH_LO <= p <= PH_HI),
 "ph_rm6h": _roll_mean(ph, 24),
 "ph_rs6h": _roll_std(ph, 24),
 "cond_rm1h": _roll_mean(cd, 4),
 "cond_rs1h": _roll_std(cd, 4),
 "cond_rm6h": _roll_mean(cd, 24),
 "cond_rs6h": _roll_std(cd, 24),
 "ph_mn1h": _roll_min(ph, 4),
 "ph_mx1h": _roll_max(ph, 4),
 "ph_mn6h": _roll_min(ph, 24),
 "ph_mx6h": _roll_max(ph, 24),
 "cond_mn1h": _roll_min(cd, 4),
 "cond_mx1h": _roll_max(cd, 4),
 "cond_mn6h": _roll_min(cd, 24),
 "cond_mx6h": _roll_max(cd, 24),
 }
 if len(ph) > 2:
 vals["d2"] = float((ph[-1] - ph[-2]) - (ph[-2] - ph[-3]))
 raw = np.asarray([[vals[c] for c in mm_cols]], dtype=np.float64)
 return mm.transform(raw).astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# LSTM surrogate (Table 5)
# ---------------------------------------------------------------------------


class SeqDS(Dataset):
 """Sequence concat(s_t, a_t): s_t is mm_cols + a_prev (1); a_t scaled → LSTM_IN_DIM."""

 def __init__(self, frame: pd.DataFrame, mm_cols: List[str], L: int):
 self.L = L
 self.mm_cols = mm_cols
 self.Xmm = frame[mm_cols].to_numpy(np.float32)
 self.aprev = frame["a_prev"].to_numpy(np.float32)
 self.phys = frame["pH_phys"].to_numpy(np.float64)
 self.a = frame["a"].to_numpy(np.int64)

 def __len__(self):
 return max(0, len(self.Xmm) - self.L - 1)

 def __getitem__(self, i: int):
 sl = slice(i, i + self.L)
 x_mm = self.Xmm[sl]
 ap = self.aprev[sl].reshape(-1, 1)
 x = np.concatenate([x_mm, ap], axis=1)
 a_t = (self.a[sl] / 10.0).astype(np.float32).reshape(-1, 1)
 xa = np.concatenate([x, a_t], axis=1)
 y = float(self.phys[i + self.L] - self.phys[i + self.L - 1])
 ph_t = float(self.phys[i + self.L - 1])
 return torch.from_numpy(xa), torch.tensor(y, dtype=torch.float32), torch.tensor(ph_t, dtype=torch.float32)


class LSTMSurrogate(nn.Module):
 def __init__(self, in_dim: int = LSTM_IN_DIM, h1: int = 256, h2: int = 128, p: float = 0.20):
 super().__init__()
 self.l1 = nn.LSTM(in_dim, h1, batch_first=True)
 self.d1 = nn.Dropout(p)
 self.l2 = nn.LSTM(h1, h2, batch_first=True)
 self.d2 = nn.Dropout(p)
 self.fc = nn.Sequential(nn.Linear(h2, 64), nn.ReLU(), nn.Linear(64, 1))

 def forward(self, x: torch.Tensor, dropout_active: bool) -> torch.Tensor:
 y, _ = self.l1(x)
 y = self.d1(y) if dropout_active else y
 y, _ = self.l2(y)
 y = self.d2(y) if dropout_active else y
 y = y[:, -1, :]
 return self.fc(y).squeeze(-1)


@torch.no_grad()
def mc_predictive_variance(model: LSTMSurrogate, x: torch.Tensor, T: int, device: torch.device) -> float:
 model.train()
 preds = []
 for _ in range(T):
 preds.append(model(x, dropout_active=True).detach().cpu().numpy())
 model.eval()
 preds = np.stack(preds, axis=0)
 return float(np.var(preds, axis=0).mean())


def train_lstm(
 train_ds: SeqDS,
 val_ds: SeqDS,
 device: torch.device,
 L: int,
 epochs: Optional[int] = None,
 patience: Optional[int] = None,
) -> Tuple[LSTMSurrogate, StandardScaler, float]:
 """Table 4 P3 + Table 5 - Huber on **standardised** ΔpH; returns σ_model (RMSE of ΔpH in pH units)."""
 epochs = epochs if epochs is not None else LSTM_EPOCHS
 patience = patience if patience is not None else LSTM_PATIENCE
 ys = [train_ds[i][1].item() for i in range(len(train_ds))]
 dph_scaler = StandardScaler().fit(np.asarray(ys, dtype=np.float64).reshape(-1, 1))

 model = LSTMSurrogate().to(device)
 opt = optim.Adam(model.parameters(), lr=LSTM_LR)
 sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 10), eta_min=1e-5)
 tr = DataLoader(train_ds, batch_size=LSTM_BATCH, shuffle=True, drop_last=True)
 va = DataLoader(val_ds, batch_size=LSTM_BATCH, shuffle=False, drop_last=False)
 loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
 best, bad, state = 1e9, 0, None
 for ep in range(epochs):
 model.train()
 for xb, yb, _ in tr:
 xb = xb.to(device)
 yb_s = torch.from_numpy(
 dph_scaler.transform(yb.numpy().reshape(-1, 1)).astype(np.float32)
 ).squeeze(-1).to(device)
 pred = model(xb, dropout_active=True)
 loss = loss_fn(pred, yb_s)
 opt.zero_grad()
 loss.backward()
 nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 opt.step()
 sched.step()
 model.eval()
 se = []
 for xb, yb, phb in va:
 xb, yb, phb = xb.to(device), yb.to(device), phb.to(device)
 pred_s = model(xb, dropout_active=False)
 dph_hat = torch.from_numpy(
 dph_scaler.inverse_transform(pred_s.detach().cpu().numpy().reshape(-1, 1)).astype(np.float32)
 ).squeeze(-1).to(device)
 ph_next_hat = torch.clamp(phb + dph_hat, 0.0, 14.0)
 ph_next = torch.clamp(phb + yb, 0.0, 14.0)
 se.append(torch.mean((ph_next_hat - ph_next) ** 2).item())
 vmse = float(np.mean(se))
 if vmse < best - 1e-7:
 best, bad, state = vmse, 0, {k: v.detach().cpu() for k, v in model.state_dict().items()}
 else:
 bad += 1
 if bad >= patience:
 break
 if state is not None:
 model.load_state_dict(state)
 model.eval()
 resids = []
 with torch.no_grad():
 for xb, yb, _ in va:
 xb = xb.to(device)
 pred_s = model(xb, dropout_active=False).cpu().numpy().ravel()
 dph_p = dph_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
 dph_t = yb.cpu().numpy().ravel()
 resids.extend((dph_p - dph_t) ** 2)
 arr = np.asarray(resids, dtype=np.float64)
 arr = arr[np.isfinite(arr)]
 sigma_model = float(np.sqrt(np.mean(arr))) if len(arr) else 0.05
 return model, dph_scaler, sigma_model


# ---------------------------------------------------------------------------
# Running reward normalization (Table 13)
# ---------------------------------------------------------------------------


class RunningRewardNorm:
 def __init__(self, win: int = 1000):
 self.buf: List[float] = []
 self.win = win

 def norm(self, r: float) -> float:
 self.buf.append(float(r))
 if len(self.buf) > self.win:
 self.buf.pop(0)
 a = np.asarray(self.buf, dtype=np.float64)
 m, s = float(a.mean()), float(a.std() + 1e-8)
 return (r - m) / s


# ---------------------------------------------------------------------------
# Environment (Sec 3-5) - no Gym dependency
# ---------------------------------------------------------------------------


class WastewaterMDP:
 """Table 8 observations + Eq. 5 hybrid dynamics + Table 13 noise (fitted σ)."""

 def __init__(
 self,
 lstm: LSTMSurrogate,
 A_T: float,
 C_T: float,
 device: torch.device,
 physics_warm: int,
 curriculum_steps: int,
 unc_p95: Optional[float],
 mc_T: int,
 mm: MinMaxScaler,
 mm_cols: List[str],
 dph_scaler: StandardScaler,
 sigma_ph: float,
 sigma_model: float,
 sigma_cond: float,
 augment_observations: bool = False,
 ):
 self.lstm = lstm
 self.A_T = A_T
 self.C_T = C_T
 self.device = device
 self.physics_warm = physics_warm
 self.curriculum_steps = curriculum_steps
 self.unc_p95 = unc_p95
 self.mc_T = mc_T
 self.mm = mm
 self.mm_cols = mm_cols
 self.dph_scaler = dph_scaler
 self.sigma_ph = sigma_ph
 self.sigma_model = sigma_model
 self.sigma_cond = sigma_cond
 self.augment_observations = augment_observations
 self.global_step = 0
 self.reward_norm = RunningRewardNorm(1000)
 self.mm_obs_idx = [mm_cols.index(c) for c in TABLE8_COLS if c != "a_prev"]
 self.ph_buf: List[float] = []
 self.cond_buf: List[float] = []
 self.do_buf: List[float] = []
 self.turb_buf: List[float] = []
 self.lstm_rows: deque = deque([[0.0] * LSTM_IN_DIM for _ in range(L_SEQ)], maxlen=L_SEQ)
 self.reset(np.random.default_rng(0))

 def reset(self, rng: np.random.Generator):
 self.t = 0
 self.prev_a = 0
 self.last_dph = 0.0
 if rng.random() < 0.5:
 self.ph = float(rng.uniform(PH_LO, PH_HI))
 else:
 self.ph = float(rng.uniform(4.0, 6.4)) if rng.random() < 0.5 else float(rng.uniform(8.6, 11.0))
 self.cond = float(rng.uniform(500.0, 1500.0))
 self.do = float(rng.uniform(4.0, 8.0))
 self.turb = float(rng.uniform(5.0, 25.0))
 self.ph_buf = [self.ph] * 96
 self.cond_buf = [self.cond] * 96
 self.do_buf = [self.do] * 96
 self.turb_buf = [self.turb] * 96
 self.hour = float(rng.uniform(0.0, 24.0))
 row = self._lstm_row(self.prev_a, 0)
 self.lstm_rows.clear()
 for _ in range(L_SEQ):
 self.lstm_rows.append(row.copy())
 return self._obs()

 def _hour(self) -> float:
 return (self.hour + self.t * 0.25) % 24.0

 def _lstm_row(self, aprev: int, action: int) -> np.ndarray:
 row_mm = mm_row_from_histories(
 self.ph_buf, self.cond_buf, self.do_buf, self.turb_buf, self._hour(), self.mm, self.mm_cols
 )
 return np.concatenate(
 [row_mm, np.array([aprev / 10.0, action / 10.0], dtype=np.float32)]
 ).astype(np.float32)

 def _obs(self) -> np.ndarray:
 row_mm = mm_row_from_histories(
 self.ph_buf, self.cond_buf, self.do_buf, self.turb_buf, self._hour(), self.mm, self.mm_cols
 )
 v = np.concatenate([row_mm[self.mm_obs_idx], np.array([self.prev_a / 10.0], dtype=np.float32)])
 return v.astype(np.float32)

 def _apply_obs_noise(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
 """Table 13 - σ on pH and conductivity mapped into MinMax-normalised feature units."""
 if not self.augment_observations:
 return obs
 o = obs.copy()
 i_ph = self.mm_cols.index("pH_raw")
 i_cd = self.mm_cols.index("cond")
 span_ph = float(self.mm.data_max_[i_ph] - self.mm.data_min_[i_ph] + 1e-9)
 span_cd = float(self.mm.data_max_[i_cd] - self.mm.data_min_[i_cd] + 1e-9)
 o[0] += float(rng.normal(0, self.sigma_ph / span_ph))
 o[6] += float(rng.normal(0, self.sigma_cond / span_cd))
 return o.astype(np.float32)

 def step(self, action: int, rng: np.random.Generator):
 action = int(action)
 ph0 = self.ph
 cur = self._lstm_row(self.prev_a, action)
 hist = list(self.lstm_rows)[-(L_SEQ - 1) :]
 xa_np = np.stack(hist + [cur], axis=0).astype(np.float32)
 xa_t = torch.from_numpy(xa_np).float().unsqueeze(0).to(self.device)

 if self.global_step <= self.physics_warm:
 ph2_clean = f_titration(self.ph, action, self.A_T, self.C_T)
 sig_ph = self.sigma_ph
 else:
 with torch.no_grad():
 pred_s = self.lstm(xa_t, dropout_active=False).item()
 dph = float(self.dph_scaler.inverse_transform(np.array([[pred_s]]))[0, 0])
 ph2_clean = float(np.clip(self.ph + dph, 0.0, 14.0))
 sig_ph = math.sqrt(self.sigma_ph**2 + self.sigma_model**2)

 ph2 = ph2_clean + float(rng.normal(0, sig_ph))
 ph2 = float(np.clip(ph2, 0.0, 14.0))
 self.cond += float(rng.normal(0, self.sigma_cond * 0.15))
 self.do += float(rng.normal(0, 0.02))
 self.turb += float(rng.normal(0, 0.15))
 self.cond = float(np.clip(self.cond, 50.0, 3000.0))
 self.do = float(np.clip(self.do, 0.0, 20.0))
 self.turb = float(np.clip(self.turb, 0.0, 200.0))

 Rc = float(PH_LO <= self.ph <= PH_HI)
 Rd = -abs(self.ph - PH_MID)
 vol = ACTION_VOLUMES_ML[action]
 Rdo = -((vol / DOSE_MAX_ML) ** 2)
 d1 = ph2 - self.ph
 Ro = -float((np.sign(d1) != np.sign(self.last_dph)) and (abs(d1) > 0.2))
 gap = min(abs(self.ph - PH_LO), abs(self.ph - PH_HI))
 Re = -float((action > self.prev_a + 2) and (self.prev_a != 0) and (gap < 0.5))
 r = W_COMP * Rc + W_DEV * Rd + W_DOSE * Rdo + W_OVER * Ro + W_ESC * Re

 if self.unc_p95 is not None and self.global_step > self.physics_warm:
 var = mc_predictive_variance(self.lstm, xa_t, self.mc_T, self.device)
 if var > self.unc_p95:
 r += W_UNC

 rn = self.reward_norm.norm(r)
 term = not (PH_HARD_LO <= ph2 <= PH_HARD_HI)
 if term:
 rn -= 5.0

 self.last_dph = d1
 self.prev_a = action
 self.ph = ph2
 self.ph_buf.append(self.ph)
 self.cond_buf.append(self.cond)
 self.do_buf.append(self.do)
 self.turb_buf.append(self.turb)
 if len(self.ph_buf) > 96:
 self.ph_buf.pop(0)
 self.cond_buf.pop(0)
 self.do_buf.pop(0)
 self.turb_buf.pop(0)
 self.lstm_rows.append(cur)
 self.t += 1
 self.global_step += 1

 done = term or (self.t >= T_MAX)
 obs = self._obs()
 obs = self._apply_obs_noise(obs, rng)
 return obs, float(rn), bool(done), {"raw_reward": r, "dyna_x": xa_np, "dyna_dph": float(ph2_clean - ph0)}

 def action_mask(self, rng: np.random.Generator) -> np.ndarray:
 m = np.ones(11, dtype=np.float32)
 if self.global_step >= self.curriculum_steps:
 return m
 dph = abs(self.ph - PH_MID)
 p_mask = 1.0 - min(dph / PH_RANGE_MASK, 1.0)
 for a in (4, 5, 9, 10):
 if rng.random() < p_mask:
 m[a] = 0.0
 if m.sum() < 1e-6:
 m[:] = 1.0
 return m


# ---------------------------------------------------------------------------
# PPO (Table 12) - discrete, masked logits
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
 def __init__(self, obs_dim: int = 13):
 super().__init__()
 layers = []
 d = obs_dim
 for w in (256, 128, 64):
 layers += [nn.Linear(d, w), nn.LayerNorm(w), nn.ReLU()]
 d = w
 self.enc = nn.Sequential(*layers)
 self.pi1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
 self.v1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
 self.logits = nn.Linear(32, 11)
 self.v = nn.Linear(32, 1)
 self.obs_rms = RunningMeanStd(obs_dim)

 def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
 xn = self.obs_rms.normalize(x)
 h = self.enc(xn)
 lh = self.pi1(h)
 logits = self.logits(lh)
 if mask is not None:
 logits = logits.masked_fill(mask < 0.5, -1e9)
 dist = torch.distributions.Categorical(logits=logits)
 v = self.v(self.v1(h)).squeeze(-1)
 return dist, v


class RunningMeanStd:
 def __init__(self, n: int, eps: float = 1e-4):
 self.mean = torch.zeros(n)
 self.var = torch.ones(n)
 self.count = eps

 def to(self, device):
 self.mean = self.mean.to(device)
 self.var = self.var.to(device)
 return self

 def update(self, x: torch.Tensor):
 bs = x.size(0)
 tot = self.count + bs
 delta = x.mean(dim=0) - self.mean.to(x.device)
 new_mean = self.mean.to(x.device) + delta * (bs / tot)
 m2 = x.var(dim=0, unbiased=False)
 new_var = (self.var.to(x.device) * self.count + m2 * bs) / tot
 self.mean, self.var, self.count = new_mean.detach(), new_var.detach(), float(tot)

 def normalize(self, x: torch.Tensor) -> torch.Tensor:
 return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + 1e-8)


DYNA_EVERY_STEPS = 500_000
DYNA_FINETUNE_EPOCHS = 10
DYNA_BUFFER = 50_000


def dyna_refresh_lstm(
 lstm: LSTMSurrogate,
 buffer: List[Tuple[np.ndarray, float]],
 dph_scaler: StandardScaler,
 device: torch.device,
 epochs: int = DYNA_FINETUNE_EPOCHS,
) -> None:
 """Table 13 - Dyna-style surrogate refresh on recent synthetic transitions."""
 if len(buffer) < 256:
 return
 tail = buffer[-DYNA_BUFFER :]
 random.shuffle(tail)
 opt = optim.Adam(lstm.parameters(), lr=3e-5)
 lstm.train()
 for _ in range(epochs):
 for k in range(0, len(tail), 64):
 batch = tail[k : k + 64]
 if len(batch) < 8:
 continue
 xa = torch.stack([torch.from_numpy(b[0]).float() for b in batch]).to(device)
 y = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
 y_s = torch.from_numpy(
 dph_scaler.transform(y.detach().cpu().numpy().reshape(-1, 1)).astype(np.float32)
 ).squeeze(-1).to(device)
 pred = lstm(xa, dropout_active=True)
 loss = nn.functional.huber_loss(pred, y_s, delta=HUBER_DELTA, reduction="mean")
 opt.zero_grad(set_to_none=True)
 loss.backward()
 opt.step()
 lstm.eval()


def ppo_train(
 env_factory,
 total_steps: int,
 rollout_len: int,
 minibatch: int,
 device: torch.device,
 lr0: float = 3e-4,
 lr1: float = 3e-5,
 warmup: int = 1500,
 physics_warm: int = 6000,
 curriculum_steps: int = 5000,
 seed: int = 42,
 ent_decay_steps: int = ENT_DECAY_STEPS,
 dyna: Optional[Tuple[List, StandardScaler, List]] = None,
 dyna_every: int = DYNA_EVERY_STEPS,
) -> ActorCritic:
 torch.manual_seed(seed)
 rng = np.random.default_rng(seed)
 policy = ActorCritic().to(device)
 policy.obs_rms.to(device)
 opt = optim.Adam(policy.parameters(), lr=lr0, eps=1e-5)

 dyna_holder: Optional[List] = None
 dph_scaler_ref: Optional[StandardScaler] = None
 dyna_buf: Optional[List] = None
 if dyna is not None:
 dyna_holder, dph_scaler_ref, dyna_buf = dyna

 env = env_factory()
 obs = env.reset(rng).astype(np.float32)
 step_count = 0

 def lr_of(s: int) -> float:
 if s < warmup:
 return lr0 * (s + 1) / max(1, warmup)
 prog = min(1.0, (s - warmup) / max(1, total_steps - warmup))
 return lr0 + (lr1 - lr0) * 0.5 * (1 + math.cos(math.pi * prog))

 def ent_of(s: int) -> float:
 prog = min(1.0, s / max(1, ent_decay_steps))
 return ENT_START + (ENT_END - ENT_START) * prog

 obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
 adv_buf, ret_buf = [], []

 while step_count < total_steps:
 obs_buf.clear()
 act_buf.clear()
 logp_buf.clear()
 rew_buf.clear()
 val_buf.clear()
 done_buf.clear()
 for _ in range(rollout_len):
 o_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
 mask = torch.from_numpy(env.action_mask(rng)).float().unsqueeze(0).to(device)
 dist, v = policy(o_t, mask)
 a = dist.sample()
 logp = dist.log_prob(a)
 obs2, r, done, info = env.step(int(a.item()), rng)
 if dyna_buf is not None and "dyna_x" in info:
 dyna_buf.append((info["dyna_x"], float(info["dyna_dph"])))
 if (
 dyna_holder is not None
 and dph_scaler_ref is not None
 and dyna_every > 0
 and step_count > 0
 and step_count % dyna_every == 0
 ):
 dyna_refresh_lstm(dyna_holder[0], dyna_buf, dph_scaler_ref, device)
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
 if step_count >= total_steps:
 break

 with torch.no_grad():
 o_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
 mask = torch.ones(1, 11, device=device)
 _, v_last = policy(o_t, mask)
 last = float(v_last.item())

 rewards = np.asarray(rew_buf, dtype=np.float64)
 values = np.asarray(val_buf + [last], dtype=np.float64)
 dones = np.asarray(done_buf, dtype=np.float64)
 adv = np.zeros_like(rewards)
 lastgaelam = 0.0
 for t in reversed(range(len(rewards))):
 nonterminal = 1.0 - dones[t]
 delta = rewards[t] + GAMMA * values[t + 1] * nonterminal - values[t]
 lastgaelam = delta + GAMMA * GAE_LAMBDA * nonterminal * lastgaelam
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
 for _ in range(PPO_EPOCHS):
 np.random.shuffle(idx)
 for start in range(0, n, minibatch):
 mb = idx[start : start + minibatch]
 if mb.size < 2:
 continue
 ob = obs_t[mb]
 ac = act_t[mb]
 dist, v = policy(ob, None)
 logp = dist.log_prob(ac)
 ratio = torch.exp(logp - logp_old[mb])
 clip_adv = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t[mb]
 surr = torch.min(ratio * adv_t[mb], clip_adv)
 vf_loss = 0.5 * torch.mean((ret_t[mb] - v) ** 2)
 ent = dist.entropy().mean()
 loss = -(surr.mean()) + 0.5 * vf_loss - ent_of(step_count) * ent
 for g in opt.param_groups:
 g["lr"] = lr_of(step_count)
 opt.zero_grad()
 loss.backward()
 nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
 opt.step()

 return policy


# ---------------------------------------------------------------------------
# Baselines (Table 15) + metrics (Table 16)
# ---------------------------------------------------------------------------


def rule_based_action(ph: float) -> int:
 if ph < PH_LO:
 return 8
 if ph > PH_HI:
 return 3
 return 0


class PIDController:
 def __init__(self):
 self.i = 0.0
 self.prev_e = 0.0

 def action(self, ph: float) -> int:
 e = PH_MID - ph
 self.i = np.clip(self.i + e, -5, 5)
 d = e - self.prev_e
 self.prev_e = e
 u = 2.0 * e + 0.5 * self.i + 0.3 * d
 if abs(u) < 0.05:
 return 0
 if u > 0:
 lv = int(np.clip(round(abs(u) * 2), 1, 5))
 return 5 + lv
 lv = int(np.clip(round(abs(u) * 2), 1, 5))
 return lv


def rollout_controller(
 ctrl,
 env: WastewaterMDP,
 rng: np.random.Generator,
 episodes: int,
) -> Dict[str, float]:
 dcrs, mpds, tcus, oecs = [], [], [], []
 for _ in range(episodes):
 obs = env.reset(rng)
 phs = [env.ph]
 doses = []
 pid = PIDController() if ctrl == "pid" else None
 for t in range(T_MAX):
 if ctrl == "rbt":
 a = rule_based_action(env.ph)
 elif ctrl == "pid":
 a = pid.action(env.ph)
 else:
 a = 0
 obs, _, done, _ = env.step(a, rng)
 doses.append(ACTION_VOLUMES_ML[a])
 phs.append(env.ph)
 if done:
 break
 ph_arr = np.asarray(phs)
 dcr = np.mean((ph_arr >= PH_LO) & (ph_arr <= PH_HI)) * 100.0
 mpd = np.mean(np.abs(ph_arr - PH_MID))
 tcu = float(np.sum(doses))
 d1 = np.diff(ph_arr)
 oec = int(np.sum((np.sign(d1[1:]) != np.sign(d1[:-1])) & (np.abs(d1[1:]) > 0.2)))
 dcrs.append(dcr)
 mpds.append(mpd)
 tcus.append(tcu)
 oecs.append(oec)
 return {
 "DCR": float(np.mean(dcrs)),
 "MPD": float(np.mean(mpds)),
 "TCU": float(np.mean(tcus)),
 "OEC": float(np.mean(oecs)),
 }


def rollout_policy(policy: ActorCritic, env: WastewaterMDP, rng: np.random.Generator, episodes: int, device):
 dcrs, tcus, oecs = [], [], []
 for _ in range(episodes):
 obs = env.reset(rng)
 phs = [env.ph]
 doses = []
 for _t in range(T_MAX):
 with torch.no_grad():
 o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
 dist, _ = policy(o, torch.ones(1, 11, device=device))
 a = int(dist.sample().item())
 obs, _, done, _ = env.step(a, rng)
 doses.append(ACTION_VOLUMES_ML[a])
 phs.append(env.ph)
 if done:
 break
 ph_arr = np.asarray(phs)
 dcr = np.mean((ph_arr >= PH_LO) & (ph_arr <= PH_HI)) * 100.0
 tcu = float(np.sum(doses))
 d1 = np.diff(ph_arr)
 oec = int(np.sum((np.sign(d1[1:]) != np.sign(d1[:-1])) & (np.abs(d1[1:]) > 0.2)))
 dcrs.append(dcr)
 tcus.append(tcu)
 oecs.append(oec)
 return {"DCR": float(np.mean(dcrs)), "TCU": float(np.mean(tcus)), "OEC": float(np.mean(oecs))}


def wilcoxon_report(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, float]:
 if len(x) < 3:
 return float("nan"), float("nan")
 stat, p = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
 return float(stat), float(p)


def run_full_pipeline(
 demo_mode: bool = True,
 device: Optional[torch.device] = None,
) -> Dict[str, object]:
 import os
 import sys

 device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
 n_ds1 = 25_000 if demo_mode else 120_000
 table2_flags: Dict[str, bool] = {k: False for k in ("DS-1", "DS-2", "DS-3", "DS-4", "DS-5")}

 try:
 from water_rdi_loaders import build_table2_mixed

 frames, table2_flags = build_table2_mixed(demo_mode=demo_mode, synth_module=sys.modules[__name__])
 ds1 = frames["DS-1"]
 ds2 = frames["DS-2"]
 ds3 = frames["DS-3"]
 ds4 = frames["DS-4"]
 ds5_hz = frames["DS-5"]
 except Exception as e:
 if isinstance(e, RuntimeError) and "WATER_TABLE2_REQUIRE_REAL is set" in str(e):
 raise
 ds1 = synth_ds1(n_ds1)
 ds2 = synth_ds2()
 ds3 = synth_ds3()
 ds4 = synth_ds4_monthly()
 ds5_hz = synth_ds5_hz(3600 if demo_mode else 7200)

 if len(ds1) < 500:
 ds1 = synth_ds1(n_ds1)
 table2_flags["DS-1"] = False

 rep = ds1.iloc[len(ds1) // 2]
 A_T, C_T = estimate_AT_CT_from_ds2(ds2, rep)
 acts = reconstruct_actions(ds1, A_T, C_T)

 if len(ds5_hz) < 200 or "pH" not in ds5_hz.columns:
 ds5_hz = synth_ds5_hz(3600 if demo_mode else 7200)
 table2_flags["DS-5"] = False
 sigma_ph, sigma_cond = estimate_sigmas_from_ds5(ds5_hz)
 ds5 = ds5_downsample_15m(ds5_hz)

 sl_tr, sl_va, sl_te = chronological_split(len(ds1))
 tr_raw, va_raw, te_raw = ds1.iloc[sl_tr], ds1.iloc[sl_va], ds1.iloc[sl_te]
 tr_f, st = preprocess_monitor(tr_raw, fit=True)
 va_f, _ = preprocess_monitor(va_raw, stats=st, fit=False)
 te_f, _ = preprocess_monitor(te_raw, stats=st, fit=False)

 tr_f = attach_actions(tr_f, acts[sl_tr])
 va_f = attach_actions(va_f, acts[sl_va])
 te_f = attach_actions(te_f, acts[sl_te])

 mm_cols = st["mm_cols"]
 tr_ds = SeqDS(tr_f, mm_cols, L_SEQ)
 va_ds = SeqDS(va_f, mm_cols, L_SEQ)
 lstm, dph_scaler, sigma_model = train_lstm(
 tr_ds,
 va_ds,
 device,
 L_SEQ,
 epochs=25 if demo_mode else LSTM_EPOCHS,
 patience=4 if demo_mode else LSTM_PATIENCE,
 )

 loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
 vars_train: List[float] = []
 lstm.train()
 with torch.no_grad():
 for k, (xb, _, _) in enumerate(loader):
 if k > 12:
 break
 xb = xb.to(device)
 vars_train.append(mc_predictive_variance(lstm, xb, 8 if demo_mode else 20, device))
 lstm.eval()
 unc_p95 = float(np.percentile(vars_train, 95)) if vars_train else None

 phys_warm = 6_000 if demo_mode else 100_000
 curr_steps = 5_000 if demo_mode else 200_000
 total_steps = 12_000 if demo_mode else 5_000_000
 rollout = 512 if demo_mode else 2048
 minib = 128 if demo_mode else 512
 dyna_every = 8_000 if demo_mode else DYNA_EVERY_STEPS

 def make_env(augment_obs: bool, eval_mode: bool):
 return WastewaterMDP(
 lstm,
 A_T,
 C_T,
 device,
 physics_warm=0 if eval_mode else phys_warm,
 curriculum_steps=0 if eval_mode else curr_steps,
 unc_p95=unc_p95,
 mc_T=8 if demo_mode else 30,
 mm=st["mm"],
 mm_cols=mm_cols,
 dph_scaler=dph_scaler,
 sigma_ph=sigma_ph,
 sigma_model=sigma_model,
 sigma_cond=sigma_cond,
 augment_observations=augment_obs and (not eval_mode),
 )

 dyna_buf: List[Tuple[np.ndarray, float]] = []
 lstm_holder = [lstm]
 policy = ppo_train(
 lambda: make_env(True, False),
 total_steps=total_steps,
 rollout_len=rollout,
 minibatch=minib,
 device=device,
 warmup=1500 if demo_mode else 10_000,
 physics_warm=phys_warm,
 curriculum_steps=curr_steps,
 seed=42,
 ent_decay_steps=15_000 if demo_mode else ENT_DECAY_STEPS,
 dyna=(lstm_holder, dph_scaler, dyna_buf),
 dyna_every=dyna_every,
 )

 rng = np.random.default_rng(0)
 n_ep = 12 if demo_mode else 80

 ppo_scores = [
 rollout_policy(policy, make_env(False, True), np.random.default_rng(i), 1, device)["DCR"]
 for i in range(n_ep)
 ]
 rbt_scores = []
 for i in range(n_ep):
 env_eval = make_env(False, True)
 rng_ep = np.random.default_rng(i)
 env_eval.reset(rng_ep)
 phs = [env_eval.ph]
 doses = []
 for _ in range(T_MAX):
 a = rule_based_action(env_eval.ph)
 _, _, done, _ = env_eval.step(a, rng_ep)
 doses.append(ACTION_VOLUMES_ML[a])
 phs.append(env_eval.ph)
 if done:
 break
 ph_arr = np.asarray(phs)
 rbt_scores.append(np.mean((ph_arr >= PH_LO) & (ph_arr <= PH_HI)) * 100.0)

 stat, p = wilcoxon_report(np.asarray(ppo_scores), np.asarray(rbt_scores), 0.0125)

 return {
 "RDI_TABLE2": RDI_TABLE2,
 "ds1": ds1,
 "ds2": ds2,
 "ds3": ds3,
 "ds4": ds4,
 "ds5_15m": ds5,
 "ds5_hz": ds5_hz,
 "lstm": lstm,
 "policy": policy,
 "unc_p95": unc_p95,
 "ppo_dcr": ppo_scores,
 "rbt_dcr": rbt_scores,
 "wilcoxon_stat": stat,
 "wilcoxon_p": p,
 "A_T": A_T,
 "C_T": C_T,
 "dph_scaler": dph_scaler,
 "sigma_ph": sigma_ph,
 "sigma_model": sigma_model,
 "sigma_cond": sigma_cond,
 "preprocess_stats": st,
 "mm_cols": mm_cols,
 "va_f": va_f,
 "tr_f": tr_f,
 "acts_full": acts,
 "table2_flags": table2_flags,
 }


if __name__ == "__main__":
 out = run_full_pipeline(demo_mode=True)
 print("Wilcoxon vs RBT: stat=", out["wilcoxon_stat"], "p=", out["wilcoxon_p"])
