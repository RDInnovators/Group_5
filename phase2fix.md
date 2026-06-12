# Phase 2 Fixes (Pre-Phase 3)

## Fix 1 — Rebuild LUT baseline around compliance band control

Problem found:
- Prior LUT was setpoint-centric (pH 7.5 with narrow dead-band), which caused unnecessary dosing of already-compliant water and severe underperformance.

Correction applied:
- LUT now controls to **compliance band**, not setpoint.
- Inside compliance band, controller does **no dosing**.

Rebuilt LUT table:

| Condition on pH | Action |
|---|---:|
| `pH < 6.0` | 10 (strong alkaline) |
| `6.0 <= pH < 6.5` | 8 (moderate alkaline) |
| `6.5 <= pH <= 8.5` | 0 (null) |
| `8.5 < pH <= 9.0` | 3 (moderate acid) |
| `pH > 9.0` | 5 (strong acid) |

Legacy setpoint-LUT mapping is retained in code as disabled/legacy helper for traceability.

### Re-run 40-episode sanity eval (measured)

- LUT (rebuilt): **97.0893970893971 ± 7.016202579984335**
- Rule-based threshold: **87.41683991683992 ± 15.74687618515269**
- Random reference: **28.612266112266116 ± 13.615169359952874**
- Null reference: **37.5 ± 49.02903378454601**

Required check:
- LUT beats random? **Yes**
- LUT beats null? **Yes**

So LUT is no longer broken.

### Rule-based threshold sanity

Rule-based logic was rechecked:
- `pH < 6.5` -> alkaline dose
- `pH > 8.5` -> acid dose
- inside band -> null

This is band-consistent (not setpoint-forcing). Measured DCR remains plausible at `87.42%` in short sanity run.

## Fix 2 — CER convention when TCU = 0

Issue:
- CER = `(DCR / TCU) * 1000` is undefined at `TCU=0`.

Convention adopted:
- If `TCU == 0`, set `CER = +inf` (infinite efficiency sentinel).
- Rationale: zero chemical usage implies the ratio-based efficiency is unbounded rather than missing; this avoids NaN holes while preserving semantics.

Applied in `rollout_metrics(...)`:
- `CER = (DCR/TCU)*1000` when `TCU > 0`
- `CER = inf` when `TCU == 0`

Convention check (executed):
- sample trajectory with `TCU=0.0` produced `CER=Infinity`.

## Status

Both requested fixes are implemented and validated.

No Phase 3 training was started.
