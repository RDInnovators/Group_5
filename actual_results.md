# Actual Results (Phase 3)

All values below are from executed runs with locked configuration.

## Locked Training Configuration
- reward_scale (training only): `0.1`
- gamma: `0.99`
- vf_coef: `0.5`
- critic: `default`
- advantage normalization: `on`
- reward form: `raw reward with signed compliance term, W_COMP=3.0`

## Fairness Note
Evaluation metrics are computed identically across all controllers from trajectory states/actions (DCR, MPD, TCU, CER, OEC, PST, PDCR, STG=NA). Training reward scaling is not used in metric computation.

## T1 Metrics (mean ± SD)
- rule_based: DCR 89.54±14.68, MPD 0.8091±0.3690, TCU 1508.88±2117.85, CER 48.7189±49.9163, OEC 0.00±0.00, PST 56.95±70.16, PDCR 0.9192±0.1474, STG NA
- pid_deadband: DCR 98.24±2.44, MPD 0.5412±0.2412, TCU 1873.44±2523.49, CER 38.6938±48.6576, OEC 0.01±0.09, PST 15.46±11.75, PDCR 0.9828±0.0299, STG NA
- lut: DCR 94.26±8.88, MPD 0.7587±0.3200, TCU 1509.12±2119.01, CER 48.7204±49.9149, OEC 0.00±0.00, PST 34.22±42.10, PDCR 0.9480±0.0914, STG NA
- null: DCR 49.30±49.86, MPD 1.3816±1.0101, TCU 0.00±0.00, CER 49.3010±49.8626, OEC 0.00±0.00, PST 245.40±236.72, PDCR 1.0000±0.0000, STG NA
- random: DCR 27.45±15.40, MPD 1.8718±0.5922, TCU 26314.48±1487.13, CER 0.0010±0.0006, OEC 35.67±26.32, PST 121.54±163.90, PDCR 0.7277±0.0207, STG NA
- ddpg: DCR 36.50±33.26, MPD 1.7939±0.7032, TCU 4065.96±1500.47, CER 32.8676±33.2418, OEC 0.00±0.00, PST 257.64±165.86, PDCR 0.6667±0.0000, STG NA
- ppo_full: DCR 71.03±21.95, MPD 1.1584±0.5927, TCU 20671.92±11761.29, CER 4.1555±9.7433, OEC 9.90±10.08, PST 80.32±102.18, PDCR 0.8832±0.0232, STG NA
- ppo_no_curriculum: DCR 70.66±21.97, MPD 0.8948±0.3579, TCU 17084.39±3002.53, CER 0.0079±0.0044, OEC 4.53±3.88, PST 86.59±115.95, PDCR 0.8041±0.0397, STG NA
- ppo_no_escalation: DCR 58.90±13.01, MPD 1.7010±0.3017, TCU 33817.82±5384.82, CER 0.0486±0.0684, OEC 12.52±6.61, PST 83.53±95.83, PDCR 0.9479±0.0226, STG NA
- ppo_neither: DCR 78.30±18.94, MPD 0.8895±0.4257, TCU 14694.35±8358.18, CER 0.0091±0.0050, OEC 7.95±5.86, PST 67.27±88.38, PDCR 0.9267±0.0253, STG NA

## T2 Metrics (mean ± SD)
- rule_based: DCR 95.05±7.16, MPD 0.7173±0.3569, TCU 714.00±1032.95, CER 54.6005±49.8127, OEC 0.02±0.14, PST 30.54±34.10, PDCR 0.9642±0.0703, STG NA
- pid_deadband: DCR 99.15±1.21, MPD 0.4481±0.2868, TCU 910.80±1230.07, CER 43.1011±49.5440, OEC 0.01±0.07, PST 11.10±5.83, PDCR 0.9921±0.0144, STG NA
- lut: DCR 97.45±4.19, MPD 0.6615±0.3307, TCU 719.10±1036.10, CER 54.5979±49.8155, OEC 0.03±0.17, PST 18.95±19.74, PDCR 0.9773±0.0430, STG NA
- null: DCR 54.50±49.92, MPD 1.3051±1.0524, TCU 0.00±0.00, CER 54.5000±49.9220, OEC 0.00±0.00, PST 222.22±236.13, PDCR 1.0000±0.0000, STG NA
- random: DCR 15.30±13.03, MPD 2.2288±0.5241, TCU 18789.08±7663.48, CER 0.0009±0.0010, OEC 38.88±23.39, PST 144.84±197.84, PDCR 0.7289±0.0292, STG NA
- ddpg: DCR 40.05±33.43, MPD 1.7654±0.7492, TCU 2047.80±750.18, CER 36.3338±33.2815, OEC 0.00±0.00, PST 257.87±148.95, PDCR 0.6667±0.0000, STG NA
- ppo_full: DCR 56.06±28.16, MPD 1.5223±0.7667, TCU 24057.72±12826.61, CER 3.9303±9.1327, OEC 20.40±20.23, PST 179.34±157.54, PDCR 0.8584±0.0562, STG NA
- ppo_no_curriculum: DCR 58.36±18.07, MPD 1.1263±0.3233, TCU 16884.81±3824.78, CER 0.0073±0.0041, OEC 12.99±6.63, PST 85.87±113.11, PDCR 0.7961±0.0332, STG NA
- ppo_no_escalation: DCR 47.12±19.16, MPD 1.9792±0.5254, TCU 33565.33±6051.25, CER 0.0484±0.0687, OEC 16.96±10.46, PST 214.90±101.47, PDCR 0.9326±0.0371, STG NA
- ppo_neither: DCR 53.73±29.43, MPD 1.5708±0.7819, TCU 21959.42±10757.78, CER 0.0061±0.0046, OEC 22.41±14.14, PST 119.51±164.83, PDCR 0.8781±0.0709, STG NA

## Tier 1 Statistics: PPO(full) vs Baselines (DCR, TCU, CER)
- PPO(full) vs rule_based [DCR]: Wilcoxon stat=12649.500, p=5.583e-51, p_bonf=1.005e-49, Cohen's d=-0.8394, bootstrap95 diff CI=[-20.4879, -16.6010]
- PPO(full) vs rule_based [TCU]: Wilcoxon stat=0.000, p=1.265e-83, p_bonf=2.276e-82, Cohen's d=1.6314, bootstrap95 diff CI=[18178.0229, 20207.1684]
- PPO(full) vs rule_based [CER]: Wilcoxon stat=3065.000, p=8.075e-76, p_bonf=1.453e-74, Cohen's d=-0.9583, bootstrap95 diff CI=[-48.6926, -40.4529]
- PPO(full) vs pid_deadband [DCR]: Wilcoxon stat=256.500, p=1.737e-80, p_bonf=3.126e-79, Cohen's d=-1.2796, bootstrap95 diff CI=[-29.0568, -25.3300]
- PPO(full) vs pid_deadband [TCU]: Wilcoxon stat=0.000, p=1.265e-83, p_bonf=2.276e-82, Cohen's d=1.5971, bootstrap95 diff CI=[17790.4496, 19821.3764]
- PPO(full) vs pid_deadband [CER]: Wilcoxon stat=15109.000, p=6.423e-49, p_bonf=1.156e-47, Cohen's d=-0.7504, bootstrap95 diff CI=[-38.6079, -30.3497]
- PPO(full) vs lut [DCR]: Wilcoxon stat=4244.000, p=2.388e-70, p_bonf=4.298e-69, Cohen's d=-1.0940, bootstrap95 diff CI=[-25.1197, -21.3483]
- PPO(full) vs lut [TCU]: Wilcoxon stat=0.000, p=1.265e-83, p_bonf=2.276e-82, Cohen's d=1.6314, bootstrap95 diff CI=[18159.2910, 20190.3281]
- PPO(full) vs lut [CER]: Wilcoxon stat=2190.000, p=5.231e-78, p_bonf=9.416e-77, Cohen's d=-0.9584, bootstrap95 diff CI=[-48.8284, -40.6218]
- PPO(full) vs ddpg [DCR]: Wilcoxon stat=4871.000, p=2.108e-71, p_bonf=3.794e-70, Cohen's d=1.1492, bootstrap95 diff CI=[31.9561, 37.1978]
- PPO(full) vs ddpg [TCU]: Wilcoxon stat=0.000, p=1.265e-83, p_bonf=2.276e-82, Cohen's d=1.4164, bootstrap95 diff CI=[15579.8336, 17622.2602]
- PPO(full) vs ddpg [CER]: Wilcoxon stat=31876.000, p=1.853e-21, p_bonf=3.336e-20, Cohen's d=-0.9466, bootstrap95 diff CI=[-31.3263, -26.0608]
- PPO(full) vs null [DCR]: Wilcoxon stat=31509.000, p=6.761e-19, p_bonf=1.217e-17, Cohen's d=0.4944, bootstrap95 diff CI=[17.9177, 25.5809]
- PPO(full) vs null [TCU]: Wilcoxon stat=0.000, p=1.265e-83, p_bonf=2.276e-82, Cohen's d=1.7576, bootstrap95 diff CI=[19667.7517, 21709.6561]
- PPO(full) vs null [CER]: Wilcoxon stat=31626.000, p=1.407e-21, p_bonf=2.533e-20, Cohen's d=-0.9711, bootstrap95 diff CI=[-49.2296, -41.0747]
- PPO(full) vs random [DCR]: Wilcoxon stat=1214.500, p=2.575e-80, p_bonf=4.635e-79, Cohen's d=1.7527, bootstrap95 diff CI=[41.3458, 45.6867]
- PPO(full) vs random [TCU]: Wilcoxon stat=25503.000, p=1.577e-30, p_bonf=2.838e-29, Cohen's d=-0.4716, bootstrap95 diff CI=[-6680.8664, -4607.1292]
- PPO(full) vs random [CER]: Wilcoxon stat=57.000, p=2.596e-83, p_bonf=4.673e-82, Cohen's d=0.4264, bootstrap95 diff CI=[3.3235, 5.0425]

## Ablation Statistics: PPO(full) vs Reduced
- PPO(full) vs ppo_no_curriculum [DCR]: Wilcoxon stat=57204.500, p=0.1086, p_bonf=1, Cohen's d=0.0193, bootstrap95 diff CI=[-1.3448, 2.0296]
- PPO(full) vs ppo_no_curriculum [TCU]: Wilcoxon stat=41522.000, p=6.633e-11, p_bonf=7.959e-10, Cohen's d=0.2933, bootstrap95 diff CI=[2515.9684, 4660.5525]
- PPO(full) vs ppo_no_curriculum [CER]: Wilcoxon stat=5338.000, p=2.779e-70, p_bonf=3.335e-69, Cohen's d=0.4257, bootstrap95 diff CI=[3.2980, 5.0343]
- PPO(full) vs ppo_no_curriculum [PDCR]: Wilcoxon stat=1259.500, p=2.271e-80, p_bonf=2.725e-79, Cohen's d=1.7324, bootstrap95 diff CI=[0.0751, 0.0831]
- PPO(full) vs ppo_no_escalation [DCR]: Wilcoxon stat=18323.000, p=2.548e-42, p_bonf=3.058e-41, Cohen's d=0.7362, bootstrap95 diff CI=[10.6759, 13.5591]
- PPO(full) vs ppo_no_escalation [TCU]: Wilcoxon stat=4396.000, p=1.497e-72, p_bonf=1.797e-71, Cohen's d=-1.2558, bootstrap95 diff CI=[-14072.8446, -12241.7445]
- PPO(full) vs ppo_no_escalation [CER]: Wilcoxon stat=13078.000, p=4.926e-53, p_bonf=5.911e-52, Cohen's d=0.4225, bootstrap95 diff CI=[3.2701, 4.9884]
- PPO(full) vs ppo_no_escalation [PDCR]: Wilcoxon stat=664.000, p=6.683e-82, p_bonf=8.02e-81, Cohen's d=-2.3681, bootstrap95 diff CI=[-0.0671, -0.0623]
- PPO(full) vs ppo_neither [DCR]: Wilcoxon stat=38516.500, p=1.327e-13, p_bonf=1.593e-12, Cohen's d=-0.3476, bootstrap95 diff CI=[-9.0990, -5.4085]
- PPO(full) vs ppo_neither [TCU]: Wilcoxon stat=34596.000, p=4.266e-18, p_bonf=5.119e-17, Cohen's d=0.4180, bootstrap95 diff CI=[4681.4307, 7226.1970]
- PPO(full) vs ppo_neither [CER]: Wilcoxon stat=7575.000, p=4.831e-65, p_bonf=5.797e-64, Cohen's d=0.4256, bootstrap95 diff CI=[3.2934, 5.0412]
- PPO(full) vs ppo_neither [PDCR]: Wilcoxon stat=2096.500, p=3.034e-78, p_bonf=3.64e-77, Cohen's d=-1.5110, bootstrap95 diff CI=[-0.0460, -0.0411]

## Hypotheses
- H1 (PPO vs baselines on DCR+TCU): **NOT SUPPORTED**
- H2 (progressive dosing helps CER/TCU/PDCR without hurting DCR): **NOT SUPPORTED**
- H3 (Tier 1 vs Tier 2 performance shift): **SUPPORTED**
- H4: **DEFERRED (no public real validation dataset with suitable pH+timestamp structure)**

PPO(full) does not beat the best baseline on Tier 1 DCR (71.03% vs pid_deadband 98.24%).

## Threats to Validity
- Simulation-only framing; no public empirical validation dataset met requirements.
- Single environment model assumptions (titration dynamics, noise model).
- Statistical conclusions depend on episode sampling distribution and OOD shift definition.

## Manuscript Fill Values
- Tier 1/Tier 2 metrics for each controller are in `phase3b_evaluation_summary.json`.
- Statistical test outputs are in `phase3c_stats.json`.
- Locked config and run provenance are in `phase3a_training_summary.json`.
