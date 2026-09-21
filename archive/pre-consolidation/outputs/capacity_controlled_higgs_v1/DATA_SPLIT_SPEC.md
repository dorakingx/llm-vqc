# HIGGS-v1 data-split specification

Deterministic, disjoint, class-stratified. Implemented in
`llm_vqc/experiments/capacity_controlled/higgs_task.py` and
`higgs_data.py`; sample IDs are global HIGGS row indices (immutable),
persisted in `run_summary.json.block_assignments`.

## Global regions (by official HIGGS row index)

| Region | Rows | Use |
|---|---|---|
| Cached pool | [0, 300,000) | streamed once from the verified 11M-row file |
| Qualification region | [0, 60,000) | classical qualification only (D1); no block, no VQC |
| Benchmark region | [60,000, 300,000) | source of the 10 blocks |
| (unused middle) | [300,000, 10,500,000) | not cached, not used |
| **External holdout** | [10,500,000, 11,000,000) | official test set; **untouched until final audit** |
| Holdout audit subset | [10,500,000, 10,520,000) | fixed 20,000-row external audit |

All three used regions (qualification, benchmark, holdout) are **mutually
disjoint**.

## The 10 benchmark blocks

- Exactly **10** blocks, each **500 train / 250 val / 2,000 protected test**
  (2,750 samples; 27,500 total).
- **Class-stratified**: each partition preserves the benchmark region's class
  ratio (drawn by popping from per-class deterministic queues).
- **Mutually disjoint**: no sample index appears in two blocks or two partitions
  (dealt by popping from shared per-class queues; verified in tests).
- **Deterministic**: `numpy.default_rng(20260719)` permutes each class once;
  blocks are dealt in order.
- Duplicate-feature rows cannot cross partitions because assignment is by unique
  row index, and a given row index goes to exactly one (block, partition).

## Integrity guarantees (tested)

- `test_blocks_mutually_disjoint_and_stratified`: 10 blocks, all IDs unique
  within and across blocks.
- `test_dev_benchmark_holdout_disjoint`: block IDs ⊂ [60k, 10.5M); holdout IDs ≥
  10.5M; empty intersection.
- `test_preprocessing_fitted_on_block_train_only`: perturbing val/test cannot
  change the fitted StandardScaler/PCA.
- `test_row_count_and_manifest`: 11,000,000 rows verified; 21 low-level + 7
  high-level; 64-hex compressed SHA-256.
