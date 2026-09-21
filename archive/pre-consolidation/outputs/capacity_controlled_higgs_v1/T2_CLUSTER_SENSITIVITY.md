# T2-v1 paired ablations — cluster-aware post-hoc reanalysis

**Post hoc. No simulation was rerun; the T2-v1 package is unchanged.** This
reanalyzes the *stored* T2 paired differences
(`outputs/capacity_controlled_t2_v1/paired_freeze_train_results.csv`,
`paired_entanglement_results.csv`) with correct clustering. Machine-readable:
`t2_cluster_sensitivity.json`. Reproduce: `python
scripts/capacity_controlled/t2_cluster_reanalysis.py`.

## Why the original T2 stats were optimistic

The T2 paired tests used n = 300 differences = **20 architectures/pairs × 5 data
splits × 3 seeds**. Those 300 are **not independent**: architectures repeat,
data splits repeat, and initialization conditions repeat. Treating them as 300
i.i.d. observations overstates precision (pseudo-replication).

## Correct clustering

Primary clusters: **data split (5)** and **architecture/pair (20)**; seeds (3)
nested. `statsmodels` is unavailable in this environment, so instead of a
mixed-effects model this uses (a) architecture-averaged and split-averaged
cluster-level t-tests, and (b) a hierarchical bootstrap that resamples splits,
then architectures, then a seed within each cell.

## Trainable vs frozen quantum angles (positive ⇒ training helps)

| Analysis | mean diff | 95% CI | p |
|---|---:|---|---:|
| naive i.i.d. (n=300) | −0.0076 | — | Wilcoxon 0.120 |
| architecture-averaged (n=20) | −0.0076 | [−0.0195, +0.0044] | 0.200 |
| split-averaged (n=5) | −0.0076 | — | 0.080 |
| hierarchical bootstrap | −0.0076 | [−0.0285, +0.0027] | CI includes 0 |

**Conclusion unchanged (QT2): no meaningful gain from training the quantum
angles.** Every cluster-aware CI includes 0; the weak lean toward *frozen* being
better is not significant under any clustering.

## Entangled vs product (positive ⇒ entanglement helps)

| Analysis | mean diff | 95% CI | p |
|---|---:|---|---:|
| naive i.i.d. (n=300) | +0.00001 | — | **Wilcoxon 0.028** |
| architecture-averaged (n=20) | +0.00001 | [−0.0081, +0.0081] | 0.999 |
| split-averaged (n=5) | +0.00001 | — | 0.998 |
| hierarchical bootstrap | +0.00001 | [−0.0111, +0.0150] | CI includes 0 |

**Key correction:** the naive Wilcoxon p = 0.028 was a **pseudo-replication
artifact**. Under correct clustering the p-value rises to ≈ 0.999 and every CI is
centered essentially on 0. **Conclusion strengthened (E2): entanglement provides
no meaningful benefit on T2**, and the earlier marginal "significance" does not
survive clustering.

## Bottom line

Both T2 ablation conclusions (QT2 no quantum-angle-training gain; E2 no
entanglement gain) **survive correct clustering**. The only change is that the
T2 entanglement effect must be reported as **null (p≈1 clustered)**, not as the
marginally-significant p=0.028 the i.i.d. test produced. The HIGGS-v1 experiment
adopts block-level clustered inference from the start so this does not recur.
