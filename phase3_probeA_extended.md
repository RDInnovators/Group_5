# Phase 3 Probe A Extended (1M steps, seed 11)

Configuration (exact Probe A settings):

- reward scale: `0.1`
- gamma: `0.99`
- value-loss coefficient: `0.5`
- critic: default
- advantage normalization: on
- seed: `11`
- steps: `1,000,000`
- device: `cuda`

Artifacts:

- Curve: `results/phase3a_singlecheck/ppo_seed11_probeA_extended_1M_curve.csv`
- Summary: `results/phase3a_singlecheck/phase3_probeA_extended_stats.json`

## 1) Explained variance over training

From summary:

- first: **0.0068**
- mean: **0.4564**
- last: **0.6351**
- max: **0.8597**
- tail mean (last 30% of updates): **0.6430**
- tail min (last 30%): **0.0826**

Interpretation:

- EV climbs from near zero to consistently positive/high values.
- It **does stabilize high** (tail mean ~0.64, last ~0.64), though not monotonic at every update.

## 2) DCR over training

From summary:

- first eval DCR: **28.95%**
- last eval DCR: **69.37%**
- last-5 eval mean DCR: **62.44%**

Curve behavior:

- Early/mid portions include declines and fluctuations.
- Once EV becomes reliably positive/high, DCR transitions upward and ends much higher than start.

## 3) Policy entropy over training

From summary:

- first eval entropy: **2.372**
- last eval entropy: **2.247**
- last-5 eval mean entropy: **2.255**

Interpretation:

- Entropy drops from near-uniform levels, indicating policy starts committing (not staying fully random).

## Plain verdict

Under extended Probe A:

- EV stabilizes high,
- DCR stops the earlier collapse pattern and rises strongly over the long diagnostic,
- entropy decreases from near-uniform.

This matches the “fix works” branch for critic learnability and directional policy improvement.  
Reached level in this single-seed 1M diagnostic: **~69% DCR (last eval), ~62% over last 5 evals**.

No full multi-seed run was launched in this step.
