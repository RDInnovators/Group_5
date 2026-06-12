# Phase 1a Transition Verification (Pre-Refactor)

## Question

What computes `pH_{t+1}` in `WastewaterMDP.step()` right now: first-principles physics (`f_titration` / Newton-Raphson) or LSTM surrogate?

## 1) Exact code path: action -> next observation

Current `step(action, rng)` flow in `WastewaterMDP`:

1. Build current sequence input `xa_t` from rolling state histories and action.
2. Branch on `self.global_step <= self.physics_warm`:
   - **If true:**  
     `ph2_clean = f_titration(self.ph, action, self.A_T, self.C_T)`  
     (physics / charge-balance / Newton-Raphson path)
   - **If false:**  
     `pred_s = self.lstm(xa_t, dropout_active=False).item()` then inverse scale to `dph`, then  
     `ph2_clean = clip(self.ph + dph, 0, 14)`  
     (LSTM surrogate path)
3. Add stochastic process noise to get `ph2`, update buffers.
4. Compute reward terms from updated/previous pH and action.
5. Return next observation vector from updated buffers.

So the transition is **hybrid by design**: physics only during warm-start; LSTM afterwards.

## 2) Runtime verification tests (executed)

### Test A: warm-start branch uses physics and does not call LSTM

- Setup: `physics_warm=10`, `lstm=RaiseIfCalledLSTM()` (throws if called), zero noise.
- Result: multiple early steps succeeded with no exception.
- Interpretation: while in warm-start window, `step()` uses physics path.

### Test B: after warm-start, LSTM is actively used

- Setup: `physics_warm=0`, `lstm=RaiseIfCalledLSTM()`.
- Run:
  - Step 1 succeeds (still `global_step <= physics_warm` at entry).
  - Step 2 throws `RuntimeError: LSTM_FORWARD_CALLED`.
- Interpretation: post-warm transition calls LSTM.

### Test C: removing LSTM (`lstm=None`) breaks post-warm transition

- Setup: `physics_warm=1`, `lstm=None`.
- Run:
  - Step 1 and 2 succeed (physics branch at entry for those steps).
  - Step 3 throws `TypeError: 'NoneType' object is not callable`.
- Interpretation: LSTM is required by current implementation after warm window.

### Test D: warm-step physics transition numerically equals `f_titration`

- Setup: warm branch active, zero noise, action=10.
- Observed:
  - `ph_before = 8.280548704009584`
  - `ph_after = 8.64588956391547`
  - `f_titration(ph_before, action, A_T, C_T) = 8.64588956391547`
  - absolute difference = `0.0`
- Interpretation: warm branch is exactly physics transition.

## 3) Plain-language finding

`WastewaterMDP` is **not** currently a pure first-principles simulator environment.

It uses first-principles titration only during warm-start and then switches to an LSTM-based surrogate transition. LSTM is therefore involved in transition dynamics in a real, required way.

## Phase 1a outcome

This is a **major mismatch** with the simulation-only framing that requires first-principles simulator transitions throughout.

Per instruction, work stops here before Phase 1b refactor until user confirmation.
