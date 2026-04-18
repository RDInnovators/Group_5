# Methodology v.01 — full walkthrough (from *Methodology v.01.docx*)

This document summarizes the **peer-review methodology chapter** for the MBRL wastewater dosing project: physical grounding, defensible action labels, and deployment-realistic evaluation. Section numbers follow the Word manuscript.

---

## 1. Overview and design philosophy

The work proposes **model-based reinforcement learning (MBRL)** for adaptive chemical dosing in industrial wastewater neutralization. Three commitments:

1. **Physical grounding** — Dynamics are not learned only from passive monitoring; they are anchored to a **first-principles acid–base titration simulator** checked against real observations.
2. **Defensible transition labels** — Dosing actions are **reconstructed** from monitoring data with a **validated change-point protocol** before training any surrogate.
3. **Deployment realism** — The policy is evaluated in the surrogate environment **and** against **held-out real pH trajectories** (sim-to-real gap), without requiring live plant access.

The pipeline is organized into **six stages** (Table 1 in the manuscript): dataset curation & action reconstruction → physics environment → MDP → RL training → evaluation & baselines → sim-to-real validation.

---

## 2. Datasets, curation, and dosing action reconstruction

### 2.1 Sources and selection (Table 2)

Five **public RDI-style** datasets are used, each with a defined role:

| ID | Role (short) |
|----|----------------|
| **DS-1** | National-scale monitoring — primary sequence for surrogate training and **action reconstruction** |
| **DS-2** | Regional industrial / hardness — **buffering parameters** (e.g. alkalinity proxies) |
| **DS-3** | Survey / effluent records — **compliance-style** labels (heuristic in practice) |
| **DS-4** | Global multi-decadal monthly series — **OOD** generalization, excluded from training |
| **DS-5** | High-frequency IoT — **noise** characterization and **sim-to-real** checks |

Criteria: sub-hourly or hourly pH where needed, industrial relevance, public license.

### 2.2 Why monitoring data alone is insufficient

Passive time series do **not** contain explicit dosing labels. Training a model on \((s_t, s_{t+1})\) without causal actions would not support RL. The methodology uses (1) a **physics simulator** and (2) **CUSUM + inverse physics** labeling (below).

### 2.3 Physics-grounded simulator

#### 2.3.1 Henderson–Hasselbalch / charge-balance titration

- Weak acid / strong base style system; **charge balance** is solved numerically (e.g. Newton–Raphson) for \([\mathrm{H}^+]\) after each dose.
- Parameters include **total alkalinity \(A_T\)**, **DIC / \(C_T\)**-style buffering, **tank volume** \(V_{\mathrm{system}}\), **reagent concentration** \(C_{\mathrm{reagent}}\).
- Discrete update: \(\mathrm{pH}_{t+1} = f_{\mathrm{titration}}(\mathrm{pH}_t, a_t, A_T, C_T, \ldots)\) with \(a_t \in \{0,\ldots,10\}\) (Table 9 in the manuscript).

#### 2.3.2 Simulator validation on real data (Sec 2.3.2)

The manuscript specifies **open-loop** comparison of the simulator to **DS-5** over many windows (e.g. **50 × 120 min** trajectories), with acceptance rules tied to prediction error (e.g. MAPE / median error thresholds). The **repository implements a shorter diagnostic** over the same idea; the full window study is manuscript-scale.

### 2.4 Action reconstruction from DS-1

#### 2.4.1 Phase 1 — CUSUM dosing detection

- **CUSUM** on pH increments; slack \(k\), threshold \(h\); merge nearby flags; discard tiny events.
- Output: candidate dosing times \(\{t^*_1,\ldots\}\).

#### 2.4.2 Phase 2 — Inverse assignment (Eq. 4)

- For each window, choose action \(a\) minimizing \(\lvert f_{\mathrm{titration}}(\mathrm{pH}_{t^*}, a, \ldots) - \mathrm{pH}_{t^*+1}\rvert\) over \(a \in \{1,\ldots,10\}\).
- If residual too large, **discard** event; non-event steps get **\(a=0\)**.
- Table 3 in the manuscript lists **reconstruction quality metrics** (event rate, discard rate, residual MAE, etc.).

### 2.5 Preprocessing pipeline (Table 4 — P1–P6)

Applied **leakage-safe**: statistics fit **only on the training split**.

| Step | Intent |
|------|--------|
| **P1** | Missing values — interpolate short gaps; bounded ffill/bfill on pH; drop rows with excessive missingness |
| **P2** | Outliers — rolling statistics + physical bounds; optional **uncertainty flags** |
| **P3** | **MinMax** on monitoring features (train min/max); **StandardScaler on \(\Delta\mathrm{pH}\)** for the LSTM target only |
| **P4** | Resample to **15 min**; DS-5 downsampled from 1 Hz; DS-4 used as OOD, not for surrogate training |
| **P5** | Rolling mean/std/**min/max** over 1 h and 6 h for pH and conductivity; velocity/acceleration; cyclic time |
| **P6** | **Chronological** 70% / 15% / 15% train / val / test |

---

## 3. Surrogate environment (LSTM)

### 3.1 Two-component environment

- Early training: **physics simulator** (warm-start).
- Later: **LSTM** approximates \(f_{\mathrm{titration}}\)-like dynamics with **stochasticity** (noise terms from data).

### 3.2 Architecture (Table 5)

- Stacked **LSTM** (e.g. 256 → 128), **dropout** (also used for **MC-Dropout** at inference).
- Sequence length **L** (e.g. 48 steps = 12 h at 15 min).
- **Huber** loss on **standardized \(\Delta\mathrm{pH}\)**; predict residual \(\Delta\mathrm{pH}\), then clip \(\mathrm{pH}\) to \([0,14]\).

### 3.3 Cross-validation vs simulator and gates (Table 6)

The manuscript lists **multiple acceptance gates** (RMSE/MAE on pH or residuals, sim-to-real error, long rollout drift, **ECE** for MC-Dropout, etc.). Implementation may realize a **subset** as automated tests; the rest are manuscript or extended experiments.

---

## 4. MDP formulation

### 4.1 State (Table 8)

**13-D** observation: pH, deviation from target, velocity, acceleration, rolling pH stats, conductivity, DO, turbidity, **previous action**, cyclic time (sin/cos), **compliance flag**.

### 4.2 Action space (Table 9)

**11 discrete actions**: null + five acid levels + five alkaline levels (geometric dose volumes).

### 4.3 Transition (Eq. 5)

Hybrid: physics step for warm-start steps; then **LSTM** prediction plus **Gaussian noise** (aleatoric + model variance).

### 4.4 Reward (Eq. 6)

**Five terms**: compliance, deviation from center, dose penalty, overshoot penalty, escalation penalty (weights in Table 10).

### 4.5 Episodes

- **\(T_{\max} = 480\)** steps (120 min at 15 min).
- **\(\gamma = 0.99\)**.
- Stratified init (in-window vs out-of-window pH).

---

## 5. RL agent (PPO)

### 5.1 Why PPO (Table 11)

Discrete actions, stability with learned dynamics — **PPO** primary; **DDPG** as comparison in some study designs.

### 5.2 PPO architecture (Table 12)

Shared encoder, policy head (softmax over 11 actions), value head; clip, entropy, GAE-λ, learning-rate schedule, rollout length, etc.

### 5.3 Progressive dosing

- **Reward** encourages small doses and smooth escalation.
- **Curriculum masking** can restrict high dose levels early in training (then full space).

### 5.4 Stability (Table 13)

**MC-Dropout** variance gating, **running reward normalization**, observation noise, **Dyna**-style surrogate refresh, hard termination outside safe pH.

---

## 6. Compliance framework (Sec 6)

Primary regulatory pH band **\[6.5, 8.5\]** (stricter jurisdictions); broader bands as secondary.

---

## 7. Evaluation (Sec 7)

### 7.1 Three questions

- **Q1** — RL vs baselines on the **test** environment.
- **Q2** — **OOD** (e.g. DS-4) under train-time scalers.
- **Q3** — **Sim-to-real** on DS-5 (critical for process-engineering reviewers).

### 7.2 Baselines (Table 15)

Rule-based threshold (RBT), PID, lookup table (LUT), DDPG (optional).

### 7.3 Tiers

- **Tier 1** — Many episodes on held-out DS-1-style conditions.
- **Tier 2** — OOD initialization from DS-4.
- **Tier 3** — Real DS-5 trajectories (manuscript describes virtual dosing + projection).

### 7.4 Metrics (Table 16)

DCR, MPD, TCU, CER, OEC, PST, PDCR, STG, etc.

### 7.5 Ablations (Table 17)

Variants removing warm-start, masking, uncertainty, LSTM temporal structure, Dyna refresh, etc.

---

## 8. Reproducibility (Sec 8)

Seeds, serialized preprocessing, code/data availability statement, checkpoints, deterministic evaluation — as required by the target journal.

---

## 9. Limitations (Table 19)

Surrogate fidelity, indirect sim-to-real, single primary control loop, reconstruction noise, reagent assumptions, etc.

---

## 10. Expected results (Sec 10)

Hypothesized numbers are **pre-registered hypotheses**, not claims until measured.

---

## Supplementary sections (11–17 in the Word file)

Notation (Table 22), related work, sensitivity analysis, statistical protocol, declarations, supplementary inventory, references — support the main narrative above.

---

*This walkthrough is a structured reading guide for **Methodology v.01.docx**; for what is implemented in code vs deferred, see `Implementation_and_Remaining_Work.md`.*
