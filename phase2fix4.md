# Phase 2 Fix 4 — Deadband PID Fairness Correction

## Change summary

Implemented a **deadband PID baseline** (standard process-control anti-chatter practice):

- Deadband region: **[6.7, 8.3]** (inside compliance band [6.5, 8.5])
- Behavior:
  - inside deadband -> action `0` (null)
  - outside deadband -> normal PID mapping to discrete actions
- Anti-windup in deadband:
  - integrator is **held** (not accumulated) while idle in deadband

For traceability:
- prior no-deadband PID is retained as `PIDDiscreteControllerLegacyNoDeadband` and
  `make_policy_pid_legacy_no_deadband(...)`.

## Before/after PID diagnostic (same setup as phase2fix3)

Setup:
- 12 episodes
- same simulator/start-state distribution as prior diagnostic

### PID before (no deadband)
- non-null action fraction: **0.4029513888888889**
- TCU mean: **8954.25**
- DCR mean: **96.48302148302149**

### PID after (deadband)
- non-null action fraction: **0.035416666666666666**
- TCU mean: **3024.1666666666665**
- DCR mean: **97.19334719334718**

Result:
- TCU drops substantially (~66% reduction from 8954 -> 3024)
- DCR remains high (slightly higher in this sample run)

## Action distribution comparison (PID)

Before no-deadband:
- `action 0` fraction: `0.5970`
- non-null spread across many actions, including frequent high doses (`action 10` fraction `0.0722`)

After deadband:
- `action 0` fraction: `0.9646`
- non-null actions much sparser; still able to apply strong correction when needed (`action 10` fraction `0.0314`)

## Out-of-band correctness trace

Deadband PID trace checks (controller-level):

Low side start (`pH=6.2`):
- actions: `[10, 10, 10, 10, 10, 10, 10, 10]`
- direction: alkaline correction upward (toward band)

High side start (`pH=8.8`):
- actions: `[5, 5, 5, 5, 5, 5, 5, 5]`
- direction: acid correction downward (toward band)

This confirms correct control direction outside band.

## Fairness checks for other non-learning baselines

Using the same 12-episode diagnostic:

- Rule-based non-null action fraction: **0.14965277777777777**  
  (null-in-band behavior present)
- LUT non-null action fraction: **0.09479166666666666**  
  (null-in-band behavior present)

So rule-based and LUT are already non-chattering relative to the old PID.

## Status

Deadband PID fairness fix is implemented and verified.

No Phase 3 training started.
