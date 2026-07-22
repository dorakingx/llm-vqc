# HIGGS data-scale qualification — pre-registered protocol

**Predeclared and committed before the qualification study runs.** Purpose:
determine *why* the HIGGS-v1 VQC stayed near the trivial baseline, by separating
**training-set size**, **input representation**, **training protocol**, and **model
family**. This is **qualification, not architecture search**. Companions:
`DATA_SPEC.md`, `MODEL_SPEC.md`, `DECISION_GATES.md`, `protocol.json`.

## 1. Data (see `DATA_SPEC.md`)

New qualification pool = official HIGGS rows **[1,000,000, 1,200,000)** — disjoint
by construction from the HIGGS-v1 development subset ([0, 60k)), the v1 benchmark
blocks (drawn from [60k, 300k)) and the **official external holdout**
([10.5M, 11M)), which remains **untouched**. Immutable IDs = global row indices.
Raw data stays gitignored.

**5 mutually disjoint qualification blocks**, each: training pool 10,000,
**fixed validation 2,000**, **fixed internal test 5,000** (17,000/block; 85,000
total). Training sets of **500 / 2,000 / 5,000 / 10,000** are **deterministic
nested prefixes** (500 ⊂ 2,000 ⊂ 5,000 ⊂ 10,000). Validation and test are fixed
across training sizes. Class-stratified; no sample crosses blocks or partitions.

**The statistical unit is the block.** Initialization and architecture seeds are
repeated measurements within a block.

## 2. Factorial design

- **Training size:** 500, 2,000, 5,000, 10,000.
- **Representation** (fit on training only, refit per training size):
  **R1** raw 21 + StandardScaler; **R2** StandardScaler + PCA(8);
  **R3** StandardScaler + PCA(16).
- **Model family** (see `MODEL_SPEC.md`): **M0** class prior, **M1** logistic
  regression, **M2** small classical MLP, **M3** fixed product-state VQC,
  **M4** fixed entangled VQC, **M5** frozen-quantum version of the exact M4
  architecture, **M6** trainable version of the exact M4 architecture.
  *M6 is definitionally identical to M4 (both are the trainable M4 architecture);
  it is computed once and reported under both labels.*
- **5 predeclared VQC architecture seeds** (5000–5004) for M3–M6.

## 3. Development-only training-protocol audit (runs first)

On **development data only** (a held-out development block, never a qualification
block's test labels), for fixed VQC architectures, grid:
lr ∈ {0.005, 0.01, 0.05} × epochs ∈ {20, 50} × weight_decay ∈ {1e-5, 1e-3} ×
early stopping ∈ {disabled, validation patience 8 with restored best checkpoint}.
**Batch size stays fixed at 16** (no correctness/memory issue requires a change).
Selection uses **validation log-loss only**; test labels are never accessed.
One common VQC training protocol is chosen **before** any qualification test
analysis and recorded in `training_protocol_selection.json`.

## 4. Metrics

**Primary: internal-test log-loss.** Secondary: AUROC, balanced accuracy,
classification error, Brier, expected calibration error (10 bins), runtime.
All model/protocol choices use **development/validation log-loss only**.

## 5. Capacity accounting (recorded per condition)

raw input dim; classical embedding params; trainable quantum params; frozen
quantum params; head params; total params; total trainable params; depth;
two-qubit gates. Expected embeddings: R1 21→4 = **88**, R2 8→4 = **36**,
R3 16→4 = **68**; plus **12** quantum and **5** head where applicable. Within a
representation, product/frozen/trainable/entangled comparisons preserve the
parameter pairing; cross-representation parameter differences are always reported.

## 6. Non-degeneracy gate (frozen — see `DECISION_GATES.md`)

A representation/size/VQC condition passes only if, at block level: median
validation AUROC ≥ 0.58; median internal-test AUROC ≥ 0.58; median internal-test
log-loss beats the class prior by ≥ 0.01; improvement positive in ≥ 4/5 blocks;
training failure rate < 5%; not driven by a single architecture seed; and
validation→test optimism gap < 0.02 log-loss. **Not weakened after outcomes.**

## 7. Conditional next stage

- **If no VQC condition passes:** do **not** run a new architecture-search matrix;
  conclude the 4-qubit/12-parameter family is not a viable performance benchmark
  for HIGGS under the tested conditions, and recommend one of: larger quantum
  parameter budget, data re-uploading, more qubits, different encoding, or a pivot
  to hardware-aware multi-objective search.
- **If ≥1 passes:** select by (1) best median validation log-loss, (2) lowest
  validation→test optimism, (3) lowest runtime, (4) lowest two-qubit count; freeze
  it; then create a **separate** preregistered HIGGS-v2 search protocol. The full
  search is **not** run in this exploratory stage.

## 8. Hardware-aware qualification

Every VQC condition is compiled to the fixed linear topology **0–1–2–3** (fixed
config, seed 20260719): transpiled two-qubit gates, SWAP count, transpiled depth,
runtime. Assess whether the HIGGS-v1 hardware-aware advantage is stable across
data size and representation. **Internal-test performance never tunes the
hardware-aware rule.**

## 9. Analysis

Learning curves vs training size; R1 vs R2 vs R3; product vs entangled; frozen vs
trainable; classical vs quantum; calibration; runtime scaling — all with
**block-aware confidence intervals** (blocks resampled as clusters). Architecture
and initialization repetitions are never treated as independent datasets.
Determined separately: does more data help; does PCA(8) lose information; does the
training protocol help; does quantum-angle training help; does entanglement help
at larger data sizes.

## 10. Integrity

No slides. No LLM API. Official final 500,000 rows untouched. Test labels never
used for any selection. Gate frozen a priori. Raw data, runs, databases and
weights never committed. T1/T2/HIGGS-v1 packages unmodified.
