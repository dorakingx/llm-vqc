# T2-v1 pre-registered protocol — `capacity_controlled_t2_v1`

**Predeclared before the full matrix.** Machine-readable mirror: `protocol.json`.
Metric/margin choices are justified by the *task audit* (`T2_TASK_AUDIT.md`,
`T1_V2_REVIEW.md`) — which characterizes the metric's dynamic range — and were
fixed before any arm-comparison or full-matrix outcome was seen. Nothing below is
revised after results.

## 0. Task and controlled space

- **Task:** `T2_digits_3_vs_8` (sklearn digits, 3 vs 8; 357 samples; PCA(10) fit on
  train only; stratified 214/72/71). Methodology-validation task, **not HEP**.
- **Controlled VQC space:** identical circuit grammar to T1-v2 (4 qubits, angle-RY
  on all wires, **no amplitude**, Z on all wires, exactly **12 trainable quantum
  params**, same `validate_controlled` and structural limits). Embed `Linear(10→4)`,
  head `Linear(4→1)`. **Total trainable params = 61** (44 + 12 + 5), fixed and
  verified for every primary candidate (analysis fails loudly otherwise).

## 1. Metric decision (audit-driven, documented — not a silent switch)

Validation **AUC saturates at 1.0** on this near-separable task (15/15 probed
circuits), so it cannot rank candidates. Therefore:

- **Selection metric (search-visible): validation log-loss (BCE)**,
  `lower_is_better=True` — discriminative even at AUC=1.0 (probed spread
  0.046–0.062). The task loss stays BCE; only the *selection* metric name changes,
  explicitly and documented.
- **Primary protected-test metric (equivalence): balanced-accuracy error**
  (1 − balanced accuracy @ 0.5), `lower_is_better=True`.
- **Secondary test metrics:** AUROC, accuracy, classification error, log-loss,
  Brier, confusion matrix, ROC, calibration.

## 2. Equivalence margin

**δ = 0.03** on balanced-accuracy error (3 percentage points of balanced accuracy).
**Rationale (fixed a priori):** classes are balanced; the 71-sample protected test
has ~0.014 resolution per sample, so the T-provisional 0.01 is *finer than one
test sample* and would make both equivalence and detection impossible; observed
cross-circuit test spread is ≈0.056–0.114, so δ=0.03 (≈2 test samples, well below
the spread) marks a practically negligible difference in classification quality.
**Sensitivity reported at δ ∈ {0.02, 0.03, 0.05}.** Not T1's RMSE margin of 0.002.

## 3. Primary matrix

- **Data-split seeds:** 0, 1, 2, 3, 4 (5). **Repetition seeds:** 0–5 (6).
- **Arms:** Controlled Random, Controlled Evolutionary, Controlled Greedy.
- **Budget:** B = 25 (fixed; never changed after outcomes).
- 5 × 6 × 3 = **90 search runs**. Within a split, all arms/seeds share the
  identical train/val/test partitions; per-split durable store; resumable.

## 4. Primary question and secondary questions

- **Primary (confirmatory):** Are Random / Evolutionary / Greedy practically
  equivalent on protected-test balanced-accuracy error within δ=0.03? (paired
  TOST + split-blocked bootstrap + split-mean cluster sensitivity.)
- **Secondary:** selection benefit (best-of-B); trainable vs frozen quantum;
  entangled vs product; searched VQC vs fair classical; efficiency (proposals /
  unique trainings / wall-clock).

## 5. Training protocol (predeclared)

Identical to T1-v2 **except the loss is BCE** (dictated by the classification
task, not a free choice): AdamW, lr 0.05, 20 epochs, batch 16, weight_decay 1e-5,
MultiStepLR ×0.5 @ {7,13,17}, CPU, float64, **no early stopping**, threshold 0.5
for accuracy-type metrics. The probe confirmed convergence (val log-loss ~0.03).
All primary arms share this protocol; selection uses validation log-loss only.

## 6. Classical baselines (per split × seed; selection on validation only)

- **C0 stratified/majority trivial** (sanity).
- **C1 logistic regression** — grid `C ∈ {0.1, 1, 10}`.
- **C2 linear SVM** — grid `C ∈ {0.1, 1, 10}`.
- **C3 RBF-SVM** — grid `C ∈ {0.1, 1, 10} × gamma ∈ {scale, 0.1, 0.01}`.
- **C4 tree-based** — RandomForest grid `n_estimators ∈ {100, 300} × max_depth ∈
  {3, 6, None}`; GradientBoosting `n_estimators ∈ {100} × max_depth ∈ {2, 3}`.
- **C5 fixed param-near-matched MLP** — `10→4→1` (49 params), preserved.
- **C6 fair classical neural architecture search** — B=25 validation-selected
  candidates over grammar: hidden ∈ {(4,),(6,),(8,),(4,3),(6,3),(8,4)}, activation
  ∈ {relu,tanh,sigmoid}, lr ∈ {0.01,0.05,0.1}, weight_decay ∈ {0,1e-5,1e-4}, AdamW,
  20 epochs. Record exact param counts; keep near 61 but do not weaken the grammar
  to force a match. Test used once after selection.

All classical selection uses validation on the **same primary metric direction**
(balanced-accuracy error), never the protected test.

## 7. Paired quantum ablations

- **A. Frozen vs trainable quantum angles** — **20 predeclared architectures**
  (fixed seeds), each evaluated over 5 splits × 3 rep-seeds, with quantum angles
  (i) trainable vs (ii) frozen, **identical initial quantum angles, identical
  classical init, identical data, only `requires_grad` on quantum weights
  differs**. Paired by (arch, split, seed). Primary test of quantum-angle value.
- **B. Product vs entangled paired circuits** — **20 predeclared entangled
  architectures** (≥1 two-qubit gate), each mapped deterministically to a
  parameter-matched **product** counterpart: every CRZ param block → RY rotation
  block (4 params each, count preserved); every free entangling block → H block
  (0 params, gate count preserved, two-qubit count → 0). Same encoding,
  measurement, classical capacity, and paired init seed. Evaluated over 5 splits ×
  3 rep-seeds; any depth/gate mismatch reported.
- **C. Selection benefit (not "algorithm necessity")** — from the primary Random
  trajectories, best-of-first-B on validation at B ∈ {1, 5, 10, 25}, reporting the
  selected circuit's protected-test metric; compared against the **distribution of
  fixed architectures** (the 20 trainable ablation-A architectures across splits/
  seeds). Report expected fixed performance, best-of-B selected performance, the
  validation-vs-test optimism gap, and the protected-test improvement from
  selection. Framed as validation-selection benefit, not proof a clever algorithm
  is needed.

## 8. Statistics

- **Primary:** paired TOST (δ=0.03) on protected-test balanced-accuracy error,
  paired by (split, seed); **split-blocked bootstrap** 95% CI of median diff;
  **split-mean cluster** TOST sensitivity (n=5 clusters). Sensitivity at δ ∈
  {0.02, 0.03, 0.05}. Non-significant difference ≠ equivalence.
- **Ablations:** paired mean/median difference, 95% CI, effect size (Cohen's dz /
  Cliff's δ), consistency across splits and architectures (sign-consistency).
- **Secondary continuity:** Kruskal–Wallis, Mann–Whitney U + Holm, Cliff's δ.
- **Power:** predeclared simulation using the probed metric SD.

## 9. Interpretation gates (no forced favorable result)

- **Search:** S1 equivalent / S2 different / S3 inconclusive.
- **Quantum-training:** QT1 trainable > frozen / QT2 no gain / QT3 inconclusive.
- **Entanglement:** E1 entangled > product / E2 no gain / E3 inconclusive.
- **Selection:** A1 best-of-B improves test / A2 little gain / A3 inconclusive.
- **Classical:** C1 VQC > fair classical / C2 classical ≥ VQC / C3 family-dependent.

No quantum-advantage claim. No HEP claim.

## 10. Integrity (binding)

No slides. No LLM API. No amplitude encoding in the primary condition. Protected
test never enters search or any selection (VQC or classical). δ, B=25, seeds,
splits, metric, and margin frozen a priori. No seed/split exclusion. Raw stores/
weights never committed. T1 v1/v2 never modified.
