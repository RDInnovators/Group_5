"""
Real public proxies for Table 2 RDI roles (*Methodology v.01*).

Sources (citable **USGS NWIS** public services; not a private “RDI compendium” API):
- **DS-1** - USGS NWIS **instantaneous** values (Large-Scale WQ Monitoring-style national surface water).
- **DS-2** - Optional **paired** CSV via ``WATER_DS2_CSV`` (``hardness_mgL``, ``conductivity_uScm``), e.g. from WQP
 exports you host locally; otherwise falls back to synthetic DS-2.
- **DS-3** - Optional CSV via ``WATER_DS3_CSV``; if unset, bundled ``data/rdi/ds3_wqp_effluent_md_proxy.csv`` (WQP **Effluent** pH, MD 2020-2021 extract) when present; else synthetic.
- **DS-4** - USGS **daily** values, aggregated to **monthly** (GEMS/Water-style multi-decadal proxy).
- **DS-5** - **KU-MWQ** (Mendeley) Excel under ``data/rdi/`` when selected or when the file is present (``auto``), else the **same NWIS IV** stream as DS-1 at native sampling (IoT proxy).

Environment
-----------
``WATER_USE_SYNTH_ONLY=1`` - skip all downloads; caller should use ``synth_ds*``.

``WATER_DS5_SOURCE`` - ``ku_mwq`` \| ``nwis`` \| ``auto`` (default ``auto``). ``auto`` uses bundled **Sensor data for 30 cm.xlsx** when present, otherwise NWIS IV.

``WATER_DS5_KU_MWQ_XLSX`` - optional path override for the KU-MWQ 30 cm workbook (must include **pH**).

``WATER_DS2_CSV`` - path to CSV with **real** columns ``hardness_mgL`` and ``conductivity_uScm`` (paired or site-level).
If unset, a bundled WQP-derived file ``data/rdi/ds2_wqp_usgsmd_ca_mg_spc_paired.csv`` (USGS-MD Ca+Mg→hardness + specific conductance on the same WQP activity) is used when present.

``WATER_DS3_CSV`` - path to a CSV with columns ``pH``, ``discharge_class``, ``effluent_type``,
``compliance_status`` (0/1) for **DS-3**.

``WATER_NWIS_SITE_DS1`` - USGS site number for DS-1 / DS-5 (default ``01491000`` Choptank River, MD).

``WATER_NWIS_SITE_DS4`` - USGS site for long **daily** record (optional; if unset, DS-4 tries DS-1’s site then **01646500** Potomac at Point of Rocks, MD, then ``WATER_NWIS_DV_FALLBACK_SITES``).

``WATER_NWIS_DV_FALLBACK_SITES`` - comma-separated USGS site numbers tried in order after the primary DS-4 site (default ``01646500``).

``WATER_TABLE2_REQUIRE_REAL`` - if ``1``/``true``/``yes``, :func:`build_table2_mixed` raises if any dataset used a synthetic fallback (offline NWIS, missing CSVs, etc.).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


USGS_IV = "https://waterservices.usgs.gov/nwis/iv/"
USGS_DV = "https://waterservices.usgs.gov/nwis/dv/"

# KU-MWQ - Mendeley Data DOI 10.17632/34rczh25kc.4 (CC BY 4.0); 30 cm sheet has pH + sensors for DS-5.
_KU_MWQ_DIR = (
 "KU-MWQ A Dataset for Monitoring Water Quality Using Digital Sensors"
)
_KU_MWQ_30CM = "Sensor data for 30 cm.xlsx"


def ku_mwq_default_xlsx() -> Path:
 """Default path to the bundled KU-MWQ 30 cm Excel file (relative to ``Water/``)."""
 return Path(__file__).resolve().parent / "data" / "rdi" / _KU_MWQ_DIR / _KU_MWQ_30CM


def _ds5_source_mode() -> str:
 """
 Return ``ku_mwq``, ``nwis``, or ``auto``.
 ``auto`` → ``ku_mwq`` if the default KU-MWQ file exists, else ``nwis``.
 """
 raw = os.environ.get("WATER_DS5_SOURCE", "").strip().lower()
 if raw in ("ku_mwq", "nwis"):
 return raw
 if raw in ("auto", ""):
 return "ku_mwq" if ku_mwq_default_xlsx().is_file() else "nwis"
 return "nwis"


def load_ds5_ku_mwq(path: Optional[Path] = None) -> pd.DataFrame:
 """
 Load **DS-5** from the KU-MWQ 30 cm workbook (pH, temperature, turbidity; ~1 row/min).

 Output columns: ``timestamp`` (UTC), ``pH``, ``temperature_C``, ``turbidity_NTU``.
 No specific conductance - downstream σ_cond uses defaults when absent.
 """
 if path is not None:
 p = Path(path).expanduser().resolve()
 elif os.environ.get("WATER_DS5_KU_MWQ_XLSX", "").strip():
 p = Path(os.environ["WATER_DS5_KU_MWQ_XLSX"].strip()).expanduser().resolve()
 else:
 p = ku_mwq_default_xlsx()
 if not p.is_file():
 raise FileNotFoundError(f"KU-MWQ DS-5 file not found: {p}")

 try:
 df = pd.read_excel(p, sheet_name=0, engine="openpyxl")
 except ImportError as e:
 raise ImportError(
 "Reading KU-MWQ .xlsx requires openpyxl (pip install openpyxl)."
 ) from e

 def _pick(*needles: str) -> str:
 for col in df.columns:
 s = str(col).strip().lower()
 if all(n in s for n in needles):
 return str(col)
 raise ValueError(f"KU-MWQ: could not find column matching {needles!r} in {list(df.columns)!r}")

 tcol = _pick("date", "time")
 tmpcol = _pick("temperature")
 phcol = None
 for col in df.columns:
 if str(col).strip().lower() == "ph":
 phcol = str(col)
 break
 if phcol is None:
 raise ValueError(f"KU-MWQ: no pH column in {list(df.columns)!r}")
 tbcol = _pick("turbidity")

 out = pd.DataFrame(
 {
 "timestamp": pd.to_datetime(df[tcol], utc=True),
 "pH": pd.to_numeric(df[phcol], errors="coerce"),
 "temperature_C": pd.to_numeric(df[tmpcol], errors="coerce"),
 "turbidity_NTU": pd.to_numeric(df[tbcol], errors="coerce"),
 }
 )
 out = out.dropna(subset=["pH"]).sort_values("timestamp").reset_index(drop=True)
 if len(out) < 200:
 raise ValueError(f"KU-MWQ DS-5: expected at least 200 rows with pH, got {len(out)}")
 return out


def _iv_to_ds5_frame(iv: pd.DataFrame) -> pd.DataFrame:
 """NWIS IV wide table → DS-5 Hz-style frame with canonical column names."""
 ds5_hz = iv.copy()
 ds5_hz["timestamp"] = pd.to_datetime(ds5_hz["timestamp"], utc=True)
 for a, b in (
 ("00400", "pH"),
 ("00010", "temperature_C"),
 ("00095", "conductivity_uScm"),
 ("63680", "turbidity_NTU"),
 ):
 if a in ds5_hz.columns and b not in ds5_hz.columns:
 ds5_hz[b] = ds5_hz[a]
 if "pH" not in ds5_hz.columns and "00400" in ds5_hz.columns:
 ds5_hz["pH"] = ds5_hz["00400"]
 return ds5_hz


def _use_synth_only() -> bool:
 return os.environ.get("WATER_USE_SYNTH_ONLY", "").strip() in ("1", "true", "True", "yes", "YES")


def _require_all_real(flags: Dict[str, bool]) -> None:
 if os.environ.get("WATER_TABLE2_REQUIRE_REAL", "").strip().lower() not in ("1", "true", "yes"):
 return
 bad = [k for k, v in flags.items() if not v]
 if bad:
 raise RuntimeError(
 "WATER_TABLE2_REQUIRE_REAL is set but synthetic fallbacks were used for: "
 + ", ".join(bad)
 + ". Fix NWIS access, set WATER_NWIS_SITE_DS1 / WATER_NWIS_SITE_DS4 / "
 "WATER_NWIS_DV_FALLBACK_SITES, and keep bundled CSVs under data/rdi/ (or unset WATER_TABLE2_REQUIRE_REAL for offline work)."
 )


def _http_get(url: str, timeout: float = 45.0) -> bytes:
 req = urllib.request.Request(url, headers={"User-Agent": "WaterMethodology/1.0 (research)"})
 with urllib.request.urlopen(req, timeout=timeout) as resp:
 return resp.read()


def _nwis_parse_iv_series(payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
 """Parse NWIS JSON ``instantaneous values`` into {parameterCd: DataFrame(timestamp, value)}."""
 root = payload.get("value", payload)
 out: Dict[str, pd.DataFrame] = {}
 for ts in root.get("timeSeries", []) or []:
 try:
 pcode = str(ts["variable"]["variableCode"][0]["value"])
 nodata = float(ts["variable"].get("noDataValue", -999999.0))
 except (KeyError, IndexError, TypeError):
 continue
 rows: List[Tuple[pd.Timestamp, float]] = []
 for block in ts.get("values", []) or []:
 for v in block.get("value", []) or []:
 try:
 val = float(v["value"])
 except (KeyError, TypeError, ValueError):
 continue
 if val == nodata or np.isnan(val):
 continue
 try:
 ts_p = pd.Timestamp(v["dateTime"])
 except (KeyError, ValueError, TypeError):
 continue
 rows.append((ts_p, val))
 if rows:
 df = pd.DataFrame(rows, columns=["timestamp", pcode]).sort_values("timestamp")
 df = df.drop_duplicates(subset=["timestamp"], keep="last")
 out[pcode] = df
 return out


def _nwis_parse_dv_series(payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
 root = payload.get("value", payload)
 out: Dict[str, pd.DataFrame] = {}
 for ts in root.get("timeSeries", []) or []:
 try:
 pcode = str(ts["variable"]["variableCode"][0]["value"])
 nodata = float(ts["variable"].get("noDataValue", -999999.0))
 except (KeyError, IndexError, TypeError):
 continue
 rows: List[Tuple[pd.Timestamp, float]] = []
 for block in ts.get("values", []) or []:
 for v in block.get("value", []) or []:
 try:
 val = float(v["value"])
 except (KeyError, TypeError, ValueError):
 continue
 if val == nodata or np.isnan(val):
 continue
 try:
 ts_p = pd.Timestamp(v["dateTime"])
 except (KeyError, ValueError, TypeError):
 continue
 rows.append((ts_p, val))
 if rows:
 df = pd.DataFrame(rows, columns=["timestamp", pcode]).sort_values("timestamp")
 df = df.drop_duplicates(subset=["timestamp"], keep="last")
 out[pcode] = df
 return out


def fetch_nwis_iv_site(
 site: str,
 period: str = "P120D",
 parameter_cd: str = "00400,00010,00095,00300,63680",
) -> pd.DataFrame:
 """
 Pull merged instantaneous values for one NWIS site.

 Default parameters: pH (00400), temp (00010), spec. conductance (00095),
 DO (00300), turbidity FNU (63680) when reported.
 """
 q = urllib.parse.urlencode(
 {"format": "json", "sites": site, "parameterCd": parameter_cd, "period": period}
 )
 raw = _http_get(f"{USGS_IV}?{q}")
 payload = json.loads(raw.decode("utf-8"))
 parts = _nwis_parse_iv_series(payload)
 if not parts:
 raise RuntimeError(f"NWIS IV returned no series for site={site!r}")
 # Outer-merge on timestamp
 base_key = next(iter(parts))
 merged = parts[base_key].rename(columns={base_key: base_key})
 for k, df in parts.items():
 if k == base_key:
 continue
 merged = merged.merge(df.rename(columns={k: k}), on="timestamp", how="outer")
 merged = merged.sort_values("timestamp").reset_index(drop=True)
 return merged


def nwis_iv_to_ds1_schema(df: pd.DataFrame) -> pd.DataFrame:
 """Map NWIS IV wide codes to DS-1 monitoring columns expected by ``preprocess_monitor``."""
 out = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], utc=True)})
 if "00400" in df.columns:
 out["pH"] = df["00400"]
 else:
 raise ValueError("NWIS IV missing pH (00400)")
 out["temperature_C"] = df["00010"] if "00010" in df.columns else np.nan
 out["conductivity_uScm"] = df["00095"] if "00095" in df.columns else np.nan
 out["DO_mgL"] = df["00300"] if "00300" in df.columns else np.nan
 out["turbidity_NTU"] = df["63680"] if "63680" in df.columns else np.nan
 out = out.sort_values("timestamp").reset_index(drop=True)
 # 15-minute regular grid (NWIS often 5-min); mean aggregate
 out = out.set_index("timestamp")
 out = out.resample("15min").mean().dropna(how="all").reset_index()
 out["hour"] = out["timestamp"].dt.hour.astype(float)
 return out


def fetch_nwis_dv_site(
 site: str,
 start: str = "1990-01-01",
 end: str = "2023-12-31",
 parameter_cd: str = "00400,00095",
) -> pd.DataFrame:
 q = urllib.parse.urlencode(
 {
 "format": "json",
 "sites": site,
 "parameterCd": parameter_cd,
 "startDT": start,
 "endDT": end,
 }
 )
 raw = _http_get(f"{USGS_DV}?{q}")
 payload = json.loads(raw.decode("utf-8"))
 parts = _nwis_parse_dv_series(payload)
 if not parts:
 raise RuntimeError(f"NWIS DV returned no series for site={site!r}")
 base_key = next(iter(parts))
 merged = parts[base_key].rename(columns={base_key: base_key})
 for k, df in parts.items():
 if k == base_key:
 continue
 merged = merged.merge(df.rename(columns={k: k}), on="timestamp", how="outer")
 merged = merged.sort_values("timestamp").reset_index(drop=True)
 return merged


def nwis_dv_to_ds4_monthly(df: pd.DataFrame) -> pd.DataFrame:
 """Resample daily NWIS to monthly medians → DS-4-like frame."""
 x = df.copy()
 x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
 x = x.set_index("timestamp").sort_index()
 # Build pH + conductance from known codes
 if "00400" not in x.columns:
 raise ValueError("NWIS DV missing 00400 (pH)")
 y = pd.DataFrame(
 {
 "pH": x["00400"],
 "conductivity_uScm": x["00095"] if "00095" in x.columns else np.nan,
 }
 )
 y = y.resample("MS").median().dropna(subset=["pH"], how="all")
 y = y.reset_index()
 if y.columns[0] != "timestamp":
 y = y.rename(columns={y.columns[0]: "timestamp"})
 return y[["timestamp", "pH", "conductivity_uScm"]]


def load_ds2_from_csv(path: str) -> pd.DataFrame:
 """Load **DS-2** regional industrial proxy from a user-supplied paired CSV."""
 df = pd.read_csv(path)
 need = {"hardness_mgL", "conductivity_uScm"}
 missing = need - set(df.columns)
 if missing:
 raise ValueError(f"DS-2 CSV missing columns: {sorted(missing)}")
 return df[["hardness_mgL", "conductivity_uScm"]].copy()


def load_ds3_from_csv(path: str) -> pd.DataFrame:
 """Load **DS-3** survey-style table from CSV."""
 df = pd.read_csv(path)
 need = {"pH", "discharge_class", "effluent_type", "compliance_status"}
 missing = need - set(df.columns)
 if missing:
 raise ValueError(f"DS-3 CSV missing columns: {sorted(missing)}")
 out = df[list(need)].copy()
 out["compliance_status"] = pd.to_numeric(out["compliance_status"], errors="coerce").fillna(0).astype(np.int64)
 return out


def _synth_all(demo_mode: bool, synth_module: Any) -> Dict[str, pd.DataFrame]:
 n = 12_000 if demo_mode else 120_000
 return {
 "DS-1": synth_module.synth_ds1(n),
 "DS-2": synth_module.synth_ds2(),
 "DS-3": synth_module.synth_ds3(),
 "DS-4": synth_module.synth_ds4_monthly(),
 "DS-5": synth_module.synth_ds5_hz(3600 if demo_mode else 7200),
 }


def build_table2_mixed(
 *,
 demo_mode: bool = True,
 synth_module: Any,
 nwis_site_ds1: Optional[str] = None,
 nwis_site_ds4: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
 """
 Load Table-2 style frames: **USGS real** for DS-1 / DS-4 when the network is reachable;
 **DS-5** from **KU-MWQ** (bundled Excel) or **NWIS IV** per ``WATER_DS5_SOURCE``;
 **CSV-backed** DS-2 / DS-3 when bundled paths or env point to files; else synthetic fallbacks.

 Returns ``(frames, flags)`` with ``flags['DS-1'] == True`` when NWIS IV data was used, etc.
 """
 flags = {k: False for k in ("DS-1", "DS-2", "DS-3", "DS-4", "DS-5")}
 if _use_synth_only():
 out = _synth_all(demo_mode, synth_module)
 _require_all_real(flags)
 return out, flags

 n = 12_000 if demo_mode else 120_000
 site1 = (nwis_site_ds1 or os.environ.get("WATER_NWIS_SITE_DS1", "01491000")).strip()
 site4_opt = (nwis_site_ds4 or os.environ.get("WATER_NWIS_SITE_DS4", "")).strip()
 period = "P60D" if demo_mode else "P365D"

 iv: Optional[pd.DataFrame] = None
 try:
 iv = fetch_nwis_iv_site(site1, period=period)
 ds1 = nwis_iv_to_ds1_schema(iv)
 flags["DS-1"] = True
 except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError):
 ds1 = synth_module.synth_ds1(n)
 flags["DS-1"] = False

 ds5_hz: Optional[pd.DataFrame] = None
 mode = _ds5_source_mode()
 ku_path = os.environ.get("WATER_DS5_KU_MWQ_XLSX", "").strip()
 ku_p = Path(ku_path).expanduser() if ku_path else None
 if mode == "ku_mwq":
 try:
 ds5_hz = load_ds5_ku_mwq(ku_p)
 flags["DS-5"] = True
 except (ImportError, FileNotFoundError, ValueError, OSError, KeyError) as e:
 if iv is not None:
 ds5_hz = _iv_to_ds5_frame(iv)
 flags["DS-5"] = True
 else:
 ds5_hz = synth_module.synth_ds5_hz(3600 if demo_mode else 7200)
 flags["DS-5"] = False
 else:
 if iv is not None:
 ds5_hz = _iv_to_ds5_frame(iv)
 flags["DS-5"] = True
 else:
 ds5_hz = synth_module.synth_ds5_hz(3600 if demo_mode else 7200)
 flags["DS-5"] = False

 sites_dv: List[str] = []
 if site4_opt:
 sites_dv.append(site4_opt)
 if site1 not in sites_dv:
 sites_dv.append(site1)
 for s in os.environ.get("WATER_NWIS_DV_FALLBACK_SITES", "01646500").split(","):
 s = s.strip()
 if s and s not in sites_dv:
 sites_dv.append(s)
 ds4 = synth_module.synth_ds4_monthly()
 flags["DS-4"] = False
 for s4 in sites_dv:
 try:
 dv = fetch_nwis_dv_site(s4, start="1995-01-01", end="2020-12-31")
 cand = nwis_dv_to_ds4_monthly(dv)
 if len(cand) >= 12:
 ds4 = cand
 flags["DS-4"] = True
 break
 except (
 urllib.error.URLError,
 urllib.error.HTTPError,
 TimeoutError,
 OSError,
 ValueError,
 KeyError,
 json.JSONDecodeError,
 RuntimeError,
 ):
 continue

 p2 = os.environ.get("WATER_DS2_CSV", "").strip()
 if not p2:
 _bundled_ds2 = Path(__file__).resolve().parent / "data" / "rdi" / "ds2_wqp_usgsmd_ca_mg_spc_paired.csv"
 if _bundled_ds2.is_file():
 p2 = str(_bundled_ds2)
 if p2:
 ds2 = load_ds2_from_csv(p2)
 flags["DS-2"] = True
 else:
 ds2 = synth_module.synth_ds2()
 flags["DS-2"] = False

 p3 = os.environ.get("WATER_DS3_CSV", "").strip()
 if not p3:
 _bundled_ds3 = Path(__file__).resolve().parent / "data" / "rdi" / "ds3_wqp_effluent_md_proxy.csv"
 if _bundled_ds3.is_file():
 p3 = str(_bundled_ds3)
 if p3:
 ds3 = load_ds3_from_csv(p3)
 flags["DS-3"] = True
 else:
 ds3 = synth_module.synth_ds3()
 flags["DS-3"] = False

 out = {"DS-1": ds1, "DS-2": ds2, "DS-3": ds3, "DS-4": ds4, "DS-5": ds5_hz}
 _require_all_real(flags)
 return out, flags


def build_table2_frames_or_synth(
 *,
 demo_mode: bool = True,
 synth_module: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
 """Backward-compatible name for :func:`build_table2_mixed`."""
 return build_table2_mixed(demo_mode=demo_mode, synth_module=synth_module)
