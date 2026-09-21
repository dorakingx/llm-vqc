# Cross-task claim audit (T1-v2 & T2-v1) — corrections carried into HIGGS-v1

Verified against the committed T1-v2 and T2-v1 machine-readable outputs. **The
historical T1/T2 reports are not modified**; corrected formulations are recorded
here and adopted going forward.

## 1. The main overclaim, corrected

**Replace:** "Search-strategy choice does not matter."

**With:** "Under the tested T1 and T2 conditions — 4 qubits, 12 trainable quantum
parameters, angle-RY encoding, and B=25 — no practically meaningful final
performance difference was established among Random, Evolutionary, and Greedy
Search."

The equivalence result is conditional on the tested capacity, encoding, budget,
tasks, and margins; it is not a universal statement about search strategies.

## 2. T1 and T2 equivalence strengths are NOT directly comparable

- T1 primary metric = **RMSE** (regression), margin **δ = 0.002 RMSE**.
- T2 primary metric = **balanced-accuracy error** (classification), margin
  **δ = 0.03**.

Different units and different margins → "T2 equivalence is stronger than T1"
(as a numeric comparison) is **not supported**. What *is* supported: each task
independently found practical equivalence at its own pre-registered margin (T1 at
0.002 but not 0.001; T2 at 0.03 and 0.02). HIGGS-v1 fixes a single margin on a
single metric (**log-loss, δ = 0.01**) used for both selection and final
inference, so its equivalence statement is internally coherent.

## 3. T1 findings are more provisional than the T2 paired findings

- **T1 entanglement** ("entanglement helps") came from comparing *one*
  hand-built product circuit (QP) against fixed entangled circuits — weak
  evidence (single structure).
- **T1 quantum-angle training** used only the single QR architecture (frozen
  0.02275 vs trained 0.02219) — a single-architecture contrast.
- **T1 selection benefit** compared one fixed circuit vs best-of-25.

T2-v1 improved these with 20 architectures / 20 paired circuits and identical
initialization. **HIGGS-v1 goes further:** ≥20 architectures × 10 blocks × ≥2
seeds with exact CRZ↔RZ product mapping and block-level clustered inference, so
the entanglement and quantum-training conclusions are properly powered and
clustered (not single-structure, not pseudo-replicated).

## 4. T2 split independence and ablation clustering caveats

- **T2 random splits reuse examples across splits** (each of the 5 splits is an
  independent stratified 60/20/20 draw of the *same* 357 samples), so split-level
  results are **not fully independent**. HIGGS-v1 uses **10 mutually disjoint
  blocks** (no sample in two blocks) so blocks are genuine independent clusters.
- **T2 n=300 paired ablations repeat architectures, examples, and initialization
  conditions.** Treating them as 300 i.i.d. observations overstates precision.
  See `T2_CLUSTER_SENSITIVITY.md`: under correct clustering, the T2 conclusions
  (QT2 no training gain, E2 no entanglement gain) **survive**, but the naive
  entanglement Wilcoxon "significance" (p=0.028) was a **pseudo-replication
  artifact** and vanishes (clustered p ≈ 0.999).

## 5. Corrected cross-task synthesis (adopted in HIGGS-v1)

| Question | T1 (regression, δ=0.002 RMSE) | T2 (classification, δ=0.03 bal-acc-err) |
|---|---|---|
| Arms practically equivalent? | yes @0.002 (not 0.001) | yes @0.03 & 0.02 |
| Quantum-angle training helps? | not established (1 arch) | no (20 archs, clustered) |
| Entanglement helps? | suggested (1 structure) — provisional | no (clustered p≈1) |
| Best-of-B selection helps? | yes | no |
| VQC vs strong classical? | analytical ≫ VQC (task-matched fit) | strong classical > VQC |

**Net:** the only robust, cross-task claim is the *conditional* search-equivalence
statement in §1. All quantum-component claims are task-dependent and, where
strongest (T2, clustered), null. HIGGS-v1 tests these on an official, non-saturated
HEP task with block-level clustering and an untouched external holdout to see
whether any of them changes on a genuinely harder problem.
