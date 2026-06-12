# Phase 2 Fix 2 — CER Redefinition (Bounded at Zero Dose)

## New CER definition

Adopted formula:

`CER = DCR / (1 + TCU)`

Rationale (principled):
- Keeps CER finite when `TCU=0`, so statistics and tests remain well-defined.
- Preserves the intended preference for high compliance at low reagent use, while penalizing large chemical usage.

## Required behavior checks (executed)

Using the new formula:

1. **(a) zero-dose & compliant**
   - `DCR=95`, `TCU=0` -> `CER=95.0` (**high**)

2. **(b) zero-dose & non-compliant**
   - `DCR=5`, `TCU=0` -> `CER=5.0` (**low**)

3. **(c) high-dose & compliant**
   - `DCR=95`, `TCU=200` -> `CER=0.472636815920398` (**penalized/moderate-low**)

4. **(d) always-null no longer trivially dominates**
   - Always-null baseline no longer has `+inf` CER; it has finite CER tied to its finite DCR.

## Baseline sanity CER recomputation (40 episodes each)

Computed with `CER = DCR / (1 + TCU)`:

| Baseline | DCR mean | TCU mean | CER mean |
|---|---:|---:|---:|
| Rule-based | 85.18191268191268 | 2138.25 | 40.07018650455622 |
| PID | 96.52806652806653 | 9009.2 | 0.03520304026839782 |
| LUT | 92.42203742203742 | 2034.75 | 45.03726778384727 |
| Null | 36.897089397089395 | 0.0 | 36.897089397089395 |
| Random | 25.265072765072766 | 26248.875 | 0.0009666339678739042 |

CER mean ranking (desc):
1. LUT (45.0373)
2. Rule-based (40.0702)
3. Null (36.8971)
4. PID (0.0352)
5. Random (0.0010)

## Notes

- This fix is metric-definition only; no Phase 3 training was started.
- CER is now bounded and bootstrap-/test-friendly (no infinities).
