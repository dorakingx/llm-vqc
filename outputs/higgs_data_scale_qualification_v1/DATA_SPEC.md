# Data specification — HIGGS data-scale qualification

Implemented in `llm_vqc/experiments/capacity_controlled/higgs_scale.py`.
Source is the already-verified official HIGGS cache (11,000,000 rows, compressed
SHA-256 `ea302c18…`). Raw data remains gitignored.

## Region map (global HIGGS row indices)

| Region | Rows | Status |
|---|---|---|
| HIGGS-v1 development subset | [0, 60,000) | previously used — **excluded** |
| HIGGS-v1 benchmark-block region | [60,000, 300,000) | previously used — **excluded** |
| **New qualification pool** | **[1,000,000, 1,200,000)** | this study (200,000 rows) |
| Official external holdout | [10,500,000, 11,000,000) | **untouched** (not read in this cycle) |

The qualification pool is disjoint from all three by construction. Its feature
SHA-256 is recorded in `dataset_manifest.json`.

## Blocks

**5 mutually disjoint blocks**, dealt from per-class deterministic queues
(seed 20260720), each:

| Partition | Size | Notes |
|---|---:|---|
| Training pool | 10,000 | ordered so prefixes are stratified |
| Validation | 2,000 | **fixed across training sizes** |
| Internal test | 5,000 | **fixed across training sizes** |

Total 17,000/block, 85,000 overall — verified all-unique (no sample crosses a
block or a partition).

## Nested training subsets

Training sets of **500 / 2,000 / 5,000 / 10,000** are **deterministic prefixes** of
the block's training pool: `train_ids[:n]`. Because the pool is built by appending
stratified increments (500, then +1,500, then +3,000, then +5,000), every prefix
is itself class-stratified and strictly nested: 500 ⊂ 2,000 ⊂ 5,000 ⊂ 10,000.
Verified in tests.

## Preprocessing

Fit on the **training subset of the given size only**, refit for every
(block, size, representation); validation and test are transformed by the fitted
objects. R1 = StandardScaler (dim 21); R2 = StandardScaler + PCA(8); R3 =
StandardScaler + PCA(16). Tests prove that perturbing validation/test values
cannot change the fitted scaler/PCA.

## Class balance

Pool positive fraction ≈ 0.528; each block's train/val/test preserves it
(≈0.528), well inside the [0.40, 0.60] sanity range.
