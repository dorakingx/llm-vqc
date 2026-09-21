# HIGGS task qualification (classical-only, development subset)

**Decision: D1 — HIGGS low-level qualifies.** No VQC was run; no subset was chosen
because a VQC performed well. Machine-readable: `task_qualification.json`,
`task_qualification_results.csv`. Reproduce: `python
scripts/capacity_controlled/qualify_higgs.py`.

## What was tested

Five classical families (logistic regression, linear SVM, RBF-SVM,
HistGradientBoosting, small MLP) on the **21 low-level** HIGGS features, using a
20,000-row subset of the qualification region (75% dev-train / 25% dev-val),
**disjoint from every benchmark block and from the external holdout**.
Preprocessing (StandardScaler inside each pipeline) is fit on dev-train only.

## Pre-registered pass criteria and outcome

| Criterion | Threshold | Observed | Pass |
|---|---|---|:--:|
| best dev-val AUROC in range | [0.65, 0.92] | **0.6729** | ✅ |
| not saturated (min classification error) | ≥ 0.05 | 0.34 | ✅ |
| dev-val log-loss above zero | ≥ 0.10 | **0.6455** | ✅ |
| class balance | [0.40, 0.60] | **0.526** | ✅ |
| ≥ 2 model families with distinct val log-loss | ≥ 2 | 5 distinct | ✅ |
| no duplicate row across train/val boundary | 0 | 0 | ✅ |

Per-model dev-validation results (`task_qualification_results.csv`):
logreg / linear-SVM AUROC ≈ 0.64–0.68, HistGB best AUROC ≈ 0.673, all with
log-loss ≈ 0.62–0.65 — **genuinely hard, non-saturated**, the opposite of T2
(where AUROC saturated at ≈ 1.0). This is exactly the regime in which validation
log-loss is discriminative and the quantum ablations have room to show an effect.

## Consequence

The HIGGS-v1 benchmark proceeds on **HIGGS 21 low-level → PCA(8)** with log-loss as
both the search-selection metric and the primary protected-test metric. The SUSY
fallback (dataset 279) was **not needed** and was not downloaded. The 7 high-level
engineered features are excluded from the primary condition (they may appear only
in a separately labeled secondary analysis).
