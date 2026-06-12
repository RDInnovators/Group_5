# Simulator Revalidation (Phase 1 Gate)

## Scope and rules followed

- Real data only (`WATER_TABLE2_REQUIRE_REAL=1`).
- No synthetic fallback used for reported metrics.
- No simulator parameter tuning to force pass/fail.
- Open-loop validation executed and reported with actual measured errors.

## 1) Real-data suitability for validating dosing-scale pH dynamics

| Dataset | Real source in this run | Date range present | Rows | pH variation profile | What it can support | What it cannot support |
|---|---|---|---:|---|---|---|
| DS-1 | NWIS IV (site 01491000, loader P60D) | 2026-04-08 -> 2026-06-07 | 5756 | pH 6.8-7.433, std 0.101, outside-window frac 0.000 | Low-variance monitoring sanity checks | Not enough excursion/event density for strong dosing-dynamics validation |
| DS-2 | Real CSV (`ds2_wqp_usgsmd_ca_mg_spc_paired.csv`) | No timestamps | 12524 | No pH field | Buffer/chemistry context (A_T/C_T estimation) | Cannot validate pH trajectory dynamics directly |
| DS-3 | Real CSV (`ds3_wqp_effluent_md_proxy.csv`) | No timestamps | 8000 | pH 2.3-10.4, std 1.116, outside-window frac 0.250 | Genuine high-variation pH sequence test of open-loop simulator error magnitude | No time axis; cannot validate time-constant realism or temporal sampling effects |
| DS-4 | NWIS DV monthly proxy | 2013-11 -> 2020-12 | 83 | pH 7.9-9.1, std 0.222, outside-window frac 0.096 | Broad-distribution/OOD context checks | Too sparse/coarse for dosing-scale dynamic validation |
| DS-5 | NWIS IV (site 01491000, mode auto->NWIS) | 2026-04-08 -> 2026-06-07 | 17259 | pH 6.8-7.5, std 0.102, outside-window frac 0.000 | High-frequency sensor noise characterization | Too flat for validating large dosing-scale pH dynamics |

Conclusion from data reality check: among currently available real datasets, only DS-3 has strong pH excursion amplitude for a genuine stress-test of simulator fit; however it lacks timestamps.

## 2) Open-loop simulator revalidation on genuinely varying real pH data

Validation target: first-principles titration simulator (Section 4.2.1) run open-loop against DS-3 pH sequence.

Run details:
- CUSUM reconstruction constants used: `k=0.0125`, `h=0.08`.
- Chemistry context: `A_T=1.2834695`, `C_T=0.5` (derived from real DS-2 + DS-1 context via code).
- DS-3 rows evaluated: `8000`.
- Reconstructed non-null events: `315` (`null_action_fraction=0.960625`).

Measured open-loop errors (actual run output):
- **MAE (pH): 2.7794962393671496**
- **RMSE (pH): 3.24686948563819**
- **MAPE (%): 43.924986113612945**
- **Median absolute error (pH): 2.773262578569452**

Target gate checked:
- Target from request: MAE < 0.25 pH.
- Observed MAE: 2.7795 pH.

## 3) Gate verdict

**FAIL**

The simulator does **not** reproduce genuinely varying real pH dynamics within the target threshold on available high-variation real data (DS-3). This gate does not pass.

Per instruction, work stops here and does not proceed to Phase 2.
