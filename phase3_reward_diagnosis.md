# Phase 3 Reward Diagnosis (Pre-Retrain)

Training-stop control status: a checkpoint watcher is active and will terminate the current Phase 3a process at the next completed-seed artifact boundary so no further seeds are started.

This report diagnoses the reward behavior without changing training code or retraining.

## Scope and artifacts

- Stats file: `results/phase3a/phase3_reward_diagnosis_stats.json`
- Seed-11 interim context: `phase3a_interim_seed11.md`
- Environment/training configuration used in diagnosis matches current Phase 3a runner (`phase3a_train.py`).

---

## 1) Reward scale trace (per-step terms)

Reward in `WastewaterMDP.step()` is computed as:

- `r = W_COMP*Rc + W_DEV*Rd + W_DOSE*Rdo + W_OVER*Ro + W_ESC*Re`
- with weights: `W_COMP=2.0`, `W_DEV=-1.0`, `W_DOSE=-0.3`, `W_OVER=-0.5`, `W_ESC=-0.1`
- then **normalized** before returning to PPO:
  - `rn = reward_norm.norm(r)` (running z-score over a 1000-step buffer)

Measured (deadband PID rollouts, 40 episodes, training-config env):

- Mean weighted compliance term: **+1.9650** per step
- Mean weighted deviation term: **+0.5212** per step
- Mean weighted dose term: **+0.0067** per step
- Mean weighted overshoot term: **+0.00016** per step
- Mean weighted escalation term: **+0.0000** per step
- Mean **raw** step reward (`r`): **+2.4930**
- Mean **normalized** step reward (`rn`, PPO objective): **+0.1588**

Plain answer to the question:

- The compliance bonus **is being applied at +2.0** (not removed).
- The very small logged training rewards are explained by **running reward normalization**: PPO does not optimize raw `r`; it optimizes normalized `rn`, which is centered/scaled online and can stay near zero even when raw rewards are ~2.5.

---

## 2) Reward vs compliance decoupling (PID vs always-null)

Compared under the current reward function over 40 matched-seed episodes:

- **Deadband PID (~97% DCR baseline)**  
  - Mean total episode **raw** reward: **1196.65**  
  - Mean total episode **normalized** reward: **76.21**

- **Always-null (known-bad policy)**  
  - Mean total episode **raw** reward: **1014.15**  
  - Mean total episode **normalized** reward: **85.93**

Critical check result:

- On **raw** reward, compliant PID > null (correct ordering).
- On **normalized** reward (the signal PPO trains on), **null > PID**.

This is the decoupling failure: the optimized objective (`rn`) can rank a poor-compliance low-action policy above a high-compliance policy.

---

## 3) Penalty balance (PID episode sums)

For deadband PID episodes:

- Sum of compliance bonus term per episode (mean): **943.20**
- Sum of dose+escalation terms per episode (mean): **3.22**
- Difference (compliance minus dose+escalation): **939.98**

Answer:

- Chemical penalties are **not** larger than compliance reward.
- Collapse is **not** explained by dose/escalation overpowering compliance in the raw reward.

---

## 4) Curriculum masking release in real training configuration

Training config:

- `curriculum_steps = 5000`
- `PPO_TOTAL_STEPS = 5,000,000`
- Expected restricted fraction over full run: `5000 / 5,000,000 = 0.001` (0.1%)

Probe results:

- Restricted-mask fraction in first 5000 steps: **0.9878**
- Restricted-mask fraction after 5000 (next 15000 tested): **0.0**

Answer:

- Masking releases correctly after warm-up.
- Agent is not unexpectedly action-restricted for a large fraction of training.

---

## Verdict: why PPO collapses

Primary cause is a **training-objective mismatch introduced by running reward normalization**:

- Raw reward favors compliant behavior (PID > null), but PPO optimizes normalized reward (`rn`), where ordering flips (null > PID).
- This can drive PPO toward a low-action / degenerate policy with poor DCR, matching the observed seed-11 collapse.

Not supported as primary causes:

- Compliance bonus missing/scaled away in raw reward: **No** (bonus is present at +2.0).
- Dose/escalation penalties dominating compliance: **No** (compliance term is orders larger).
- Curriculum never releasing: **No** (release is functioning as configured).
