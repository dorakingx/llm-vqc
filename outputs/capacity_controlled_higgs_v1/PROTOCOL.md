# HIGGS-v1 pre-registered protocol — `capacity_controlled_higgs_v1`

**Predeclared and committed BEFORE the first scientific VQC run.** Dataset passed
qualification (D1, `TASK_QUALIFICATION.md`). Machine-readable mirror:
`protocol.json`; companions `SPACE_SPEC.md`, `DATA_SPLIT_SPEC.md`,
`STATISTICAL_ANALYSIS_PLAN.md`. Nothing below is revised after partial outcomes.

## 0. Dataset & feature condition (frozen)

UCI **HIGGS** (id 280, DOI 10.24432/C5V312, CC BY 4.0; SHA-256 `ea302c18…`,
11,000,000 rows verified). **Primary condition uses only the 21 low-level
features**, reduced to **PCA(8)** (fit on block-train only). The 7 high-level
engineered features are excluded from the primary condition (secondary-analysis
only, separately labeled). The final **500,000 rows are the external holdout**;
untouched until §7.

## 1. Data blocks (frozen; see `DATA_SPLIT_SPEC.md`)

**10 mutually disjoint blocks**, each **500 train / 250 val / 2,000 protected
internal-test**, class-stratified, deterministic (seed 20260719), from the
benchmark region (pool rows [60,000; 300,000)). No sample in two blocks or two
partitions. The qualification region (pool rows [0; 60,000)) and the external
holdout are disjoint from all blocks. Immutable sample IDs = global HIGGS row
indices (persisted in `run_summary.json.block_assignments`). **The block is the
primary statistical cluster; seeds are repeated observations within a block.**

## 2. Preprocessing (train-only, per block)

Per block: fit `StandardScaler` on that block's **training** partition, then
`PCA(8)` on the scaled training partition; transform val/test with the fitted
objects. Never fit globally. The controlled model receives exactly **8** features.

## 3. Controlled VQC (frozen; see `SPACE_SPEC.md`)

4 qubits, angle-RY on all wires (**no amplitude**), Z on all wires, **exactly 12
trainable quantum params**, embed `Linear(8→4)`=36, head `Linear(4→1)`=5 →
**total 53 trainable params** (verified per candidate; analysis fails loudly
otherwise). Circuit grammar/validator/limits inherited unchanged from T1/T2.

## 4. Metrics & equivalence margin (frozen)

- **Selection metric = validation log-loss** (lower better).
- **Primary protected-test metric = protected-test log-loss** (same metric for
  selection and final inference).
- **Secondary:** AUROC, accuracy, balanced accuracy, classification error, Brier,
  calibration, confusion matrix.
- **Primary equivalence margin = 0.01 absolute log-loss.** Sensitivity at
  {0.005, 0.010, 0.020}. Not changed after outcomes.

## 5. Training protocol (frozen)

AdamW, lr 0.05, 20 epochs, batch 16, weight_decay 1e-5, MultiStepLR ×0.5 @
{7,13,17}, **BCE loss**, CPU, float64, no early stopping, threshold 0.5 for
accuracy-type metrics. Explicit init `explicit_kaiming_uniform_qc_v1` (quantum
angles U(−π,π)). Identical for every primary arm.

## 6. Primary matrix & ablations (frozen)

- **Matrix:** 10 blocks × repetition seeds {0,1,2} × arms {Random, Evolutionary,
  Greedy} × **B=25** = **90 search runs**. All arms in a block share data +
  preprocessing. Resumable. Not resized after partial outcomes.
- **Classical baselines** (per block × seed, on the block PCA(8) features): C0
  majority, C1 logreg, C2 linear-SVM (calibrated), C3 RBF-SVM (calibrated), C4
  HistGradientBoosting, C5 RandomForest, C6 fixed MLP (8→5→1), C7 fair NAS (B=25,
  val-selected). Calibration uses train-internal CV only; test used once.
- **Freeze vs train:** 20 predeclared architectures × 10 blocks × 2 seeds {0,1},
  each trained trainable vs frozen with **identical initial quantum angles,
  classical init, minibatch order, preprocessing, architecture** (only quantum
  `requires_grad` differs).
- **Entangled vs product:** 20 predeclared entangled architectures (≥1 CRZ, no
  free entangling) × 10 blocks × 2 seeds, each mapped by the exact **CRZ(θ)→RZ(θ)
  on target** rule (1 param→1 param, gate/layer count preserved, two-qubit→0;
  depth diff recorded), paired identical init.
- **Selection benefit:** best-of-B ∈ {1,5,10,25} from Random trajectories vs the
  distribution of the 20 fixed (trainable) architectures.
- **Hardware-aware** secondary: transpile every candidate to linear topology
  0–1–2–3 (fixed config, seed 20260719); hardware-aware selection rule = best
  validation log-loss → retain within 0.01 log-loss → fewest transpiled two-qubit
  → tie transpiled depth → tie structural hash. Evaluate performance-selected and
  hardware-aware-selected on protected test; neither feeds back into search.

## 7. External holdout (frozen procedure)

After the protocol is committed, all 90-block searches complete, all analyses are
frozen, and the consensus rule is fixed: select the **architecture family with the
best aggregate validation-log-loss rank across the 10 blocks**, retrain it on a
preregistered combined training set, and evaluate **once** on the fixed 20,000-row
subset of the official final 500,000 holdout. No design decision changes based on
the result.

## 8. Statistics (see `STATISTICAL_ANALYSIS_PLAN.md`)

Primary inference unit = **data block**. Block-averaged paired arm contrasts →
**TOST (δ=0.01)** on the 10 block-level contrasts; **block bootstrap** (resample
whole blocks); cluster-averaged CIs; seeds nested within block; architecture
crossed with block for ablations. Never report the seed/architecture combination
count as the sample size.

## 9. Decision gates (no forced favorable result)

Task D1/D2/D3; Search S1/S2/S3; Quantum-training QT1/QT2/QT3; Entanglement
E1/E2/E3; Selection A1/A2/A3; Hardware H1/H2/H3; Classical C1/C2/C3. No
quantum-advantage claim. No HEP-domain claim beyond "official HEP classification
benchmark used for methodology."

## 10. Stop conditions

Stop (no full quantum benchmark) if: neither HIGGS nor SUSY qualifies (D3); the
loader cannot verify 11,000,000 rows; blocks are not mutually disjoint; or the
external holdout would be touched before the design is frozen.

## 11. Integrity (binding)

No slides. No LLM API. No amplitude encoding in the primary condition. Protected
test and external holdout never enter search or any selection. δ, B=25, blocks,
seeds, metric, margin, grammar, and 53-param capacity frozen a priori. Raw
HIGGS/SUSY data, SQLite DBs, weights never committed. T1/T2 packages untouched.
