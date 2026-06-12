# Phase 3 Learning-Dynamics Diagnosis (No Retraining)

This report investigates learning dynamics only. No training code or reward code was changed in this step.

Supporting data file:

- `results/phase3a_singlecheck/phase3_learning_diagnosis_stats.json`

## 1) Unwinnable episodes hypothesis

Test:

- 50 episodes (seeds 700..749)
- For each episode: record start pH and whether deadband PID achieves `>=80%` in-band over episode
- Episode length limit: `T_MAX = 480`

Results:

- Winnable (`PID DCR >= 80%`): **50/50 (100%)**
- Unwinnable: **0/50 (0%)**
- Start pH range in this sample: **4.398 to 10.964** (mean 7.986)
- Starts `<4` or `>11`: **0 episodes**
- Edge starts (`<=4.05` or `>=10.95`): **1 episode**
  - PID reached band in **23 steps** (well below 480-step limit)
- Hardest sampled out-of-band episode:
  - start pH **4.398**
  - PID first reached band at step **33**
  - PID DCR **93.14%**

Plain conclusion:

- In the tested distribution, episodes are **not physically unwinnable** within horizon.
- “Helplessness from unwinnable starts” is **not supported** by this check.
- Also, this environment reset configuration does not sample starts `<4` or `>11` in practice.

## 2) Signed-reward landscape on hard out-of-band start

Selected hard episode seed:

- Seed **717**, start pH **4.398**, PID reaches band at step **33**

Cumulative episode reward (locked signed/raw reward):

- Always-null: **53.40**
- Always max correct-direction dose: **1743.10**
- Deadband PID: **1652.52**

Same episode DCR:

- Always-null: **0.0%**
- Max-correct: **93.14%**
- PID: **93.14%**

Plain conclusion:

- Active dosing toward compliance earns **much higher** cumulative reward than doing nothing, even in an initially out-of-band episode before full recovery.
- This check does **not** indicate a missing directional gradient in the signed reward.

## 3) Current PPO configuration and unusual settings

From current code path:

- Learning rate: **3e-4 -> 3e-5** (cosine decay, 10k warmup)
- Entropy coefficient: **0.01 -> 0.001** (decay over 4,000,000 steps)
- Clip range: **0.2**
- Rollout length (`n_steps`): **2048**
- Minibatch: **512**
- PPO epochs: **4**
- Gamma: **0.99**
- GAE lambda: **0.95**
- Max grad norm: **0.5**
- Network:
  - Encoder MLP: **256, 128, 64**
  - Policy head: **64, 32, 11 logits**
  - Value head: **64, 32, 1**

Assessment:

- These are generally standard PPO values for discrete control.
- One potentially notable point (not necessarily wrong): entropy decay is slow (4M steps), so exploration pressure stays relatively high for a long time.
- Nothing here is an obvious pathological outlier by itself.

## 4) Entropy collapse check from short run

Requested early-collapse diagnosis from short run:

- Short run did **not** log entropy per checkpoint, so exact “early collapse timing” cannot be measured directly from logged curve.

Proxy diagnostic using saved final policy (`ppo_seed11_short.pt`) vs fresh init on 2,000 sampled states:

- Max entropy for 11 actions: **2.398**
- Initial-policy mean entropy: **2.395**
- Final-policy mean entropy: **2.386**
- Final max-action-prob mean: **0.124** (p90: 0.126)

Plain conclusion:

- Policy is still near high-entropy / non-deterministic after the short run.
- No evidence of premature near-deterministic collapse.

## Overall diagnosis

- The leading unwinnable-episodes hypothesis is **not supported** in this sampled start distribution.
- The signed reward appears to provide a strong action gradient (`max-correct/PID >> null`) on a hard out-of-band case.
- PPO did not improve in the short run, but this currently does **not** appear to be explained by unwinnable starts, missing reward gradient, or early entropy collapse.
- Most likely remaining issues are in optimization dynamics / credit assignment / training setup interactions rather than reward sign/ordering.
