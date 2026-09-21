# HIGGS-v1 statistical analysis plan (pre-registered)

The **data block is the primary inference unit.** Repetition seeds are repeated
observations *within* a block, not independent dataset replicates. The
seed/architecture combination count is **never** reported as the sample size.

## 1. Primary — search-arm equivalence (confirmatory)

- Pair arms by (block, seed) cell (arms share the block's data/preprocessing).
- **Average over seeds within each block** → one block-level paired contrast per
  arm pair → **n = 10 block-level contrasts**.
- **Paired TOST** on the 10 block-level contrasts at **δ = 0.01 log-loss**;
  equivalence iff the 90% CI of the mean block-level contrast ⊂ (−δ, +δ).
- **Block bootstrap**: resample the 10 whole blocks with replacement (2,000
  reps) → 95% CI of the mean/median block-level contrast.
- Sensitivity at δ ∈ {0.005, 0.010, 0.020}.
- Secondary continuity (exploratory): Kruskal–Wallis / Mann–Whitney U + Holm /
  Cliff's δ on block-level medians.
- A non-significant difference test is **not** treated as equivalence.
- `statsmodels` is unavailable in this environment; the block-level TOST + block
  bootstrap **are** the cluster-aware analysis (a mixed-effects model with random
  block intercepts would be the equivalent parametric form and is noted as the
  intended sensitivity model).

## 2. Paired quantum ablations (freeze/train, product/entangled)

- Clusters: **block (10)** and **architecture/pair (20)**; seeds (2) nested.
- Report: architecture-averaged effect (n=20 cluster means), block-averaged effect
  (n=10 cluster means), and a **hierarchical bootstrap** (resample blocks, then
  architectures, then a seed within each cell) → cluster-aware 95% CI.
- Report total pair count, #blocks, #architectures, #seeds, and the effective
  clustered CI — not the raw pair count as n.
- Effect size: Cohen's dz on cluster means and sign-consistency across clusters.

## 3. Selection benefit

Best-of-B ∈ {1,5,10,25} from Random trajectories (validation-selected circuit's
protected-test log-loss), aggregated at the block level; compared against the
fixed-architecture distribution (20 trainable ablation architectures). Report
expected fixed log-loss, best-of-B log-loss, validation optimism gap, and
protected-test selection gain. Framed as validation-selection benefit only.

## 4. Hardware-aware secondary

Per run: performance-selected vs hardware-aware-selected candidate (rule in
`PROTOCOL.md` §6) on protected-test log-loss and transpiled two-qubit count.
Per arm: Pareto front over (test log-loss, transpiled two-qubit); **hypervolume**
with predeclared reference point (log-loss 1.0, transpiled two-qubit 40). Compare
arms by Pareto coverage and hypervolume.

## 5. Classical comparison

Block-level median protected-test log-loss for each classical baseline (C0–C7) vs
the best VQC arm. Report family-dependence explicitly; no quantum-advantage
wording — only "outperformed / did not outperform / equivalent within margin /
inconclusive under this protocol."

## 6. Power

Post-run simulation using the observed block-level paired-diff SD; report
P(conclude equivalence via TOST δ=0.01) at true block-level diffs {0, 0.005,
0.01, 0.02} for n=10 blocks.

## 7. Decision-gate mapping

S1 iff all arm pairs TOST-equivalent at δ=0.01 (block-level); S2 iff any block-level
CI excludes (−δ,δ) with a consistent sign; else S3. QT1/E1 iff the cluster-aware
CI excludes 0 in the beneficial direction; QT2/E2 iff CI includes 0 with |effect|
< δ; else QT3/E3. A1 iff best-of-25 beats fixed at block level; H1 iff one arm's
Pareto hypervolume dominates; C1/C2/C3 per §5.
