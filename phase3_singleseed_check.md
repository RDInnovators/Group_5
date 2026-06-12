# Phase 3 Single-Seed PPO Health Check (Locked Reward)

Reward configuration used in this check (unchanged/locked):

- Raw reward (no running normalization in training signal)
- Signed compliance term (`Rc=+1` in-band, `Rc=-1` out-of-band)
- `W_COMP=3.0`, `W_DEV=-1.0`, `W_DOSE=-0.3`, `W_OVER=-0.5`, `W_ESC=-0.1`

## Step A — Environment and device check

Command run:

`python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"`

Output:

- `CUDA available: True`
- `Device count: 2`
- `Device name: NVIDIA RTX A4000`

Code placement check:

- `phase3a_train.py` uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- PPO model and tensors are moved with `.to(device)` in rollout/eval/training paths.

Plain statement:

- As written, PPO training runs on **GPU** when CUDA is available.
- No code change is needed to enable GPU for this test.

## Step B — Short single-seed PPO run (seed 11)

Run settings:

- Seed: `11`
- Steps: `400,000` (short health-check budget)
- Eval every: `50,000` steps
- Eval episodes per checkpoint: `20`
- Device used: `cuda`

Artifacts:

- Curve: `results/phase3a_singlecheck/ppo_seed11_short_curve.csv`
- Model: `results/phase3a_singlecheck/ppo_seed11_short.pt`
- Summary: `results/phase3a_singlecheck/ppo_seed11_short_summary.json`

## Step C — Directional signal

Measured from curve:

- Start eval DCR: **32.18%**
- End eval DCR: **23.90%**
- Final-window mean DCR (last 5 eval points): **26.34%**

Trend verdict:

- DCR is **declining**, not rising. This short run does **not** show learning in the desired direction.

Comparison context:

- Random reference: ~29%
- This run final-window PPO: ~26%
- LUT baseline: ~90%
- PID baseline: ~97%

Plain verdict:

- Under the locked reward, this short single-seed PPO run is still **stuck low / collapsing directionally**, not yet heading toward the competitive range.

## Status

This was a health check only. No full multi-seed run was launched.
