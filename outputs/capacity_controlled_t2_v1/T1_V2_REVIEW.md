# T1-v2 conclusion audit (for the T2-v1 cycle)

Recomputed from the committed T1-v2 machine-readable outputs (not from the
report prose). The T1-v2 output package is **not** modified; the sensitivity
analysis below is new and **post hoc**, kept in this T2 package.

## Recomputed headline (confirmed)

| Check | Recomputed value | Status |
|---|---|---|
| Primary candidates evaluated | 1,997 | ✅ |
| Unique total-param counts | {105} | ✅ every candidate exactly 105 |
| Completed search runs | 90 | ✅ |
| Invalid proposals (sum) | 0 | ✅ |
| Failed proposals (sum) | 0 | ✅ |
| Protected-test median — Random | 0.01775 | ✅ |
| Protected-test median — Evolutionary | 0.01715 | ✅ |
| Protected-test median — Greedy | 0.01689 | ✅ |
| Paired TOST @δ=0.002 | all 3 pairs equivalent | ✅ |
| Sensitivity @δ=0.001 | no pair equivalent | ✅ |
| A4 Gaussian-fit test median | 0.00116 | ✅ (≪ VQC) |

## Strongly supported (carried forward)

- **Legacy capacity confound** — legacy candidates spanned 24–22,609 params;
  amplitude-encoding (largest classical embeddings) systematically scored best.
- **Random advantage disappears under capacity control** — at fixed 105 params,
  Random is no better than (and on validation slightly worse than) the others.
- **Practical equivalence at δ=0.002** — all three arm pairs equivalent by paired
  TOST across 3 splits × 10 seeds; robust (see nesting sensitivity below).
- **No equivalence at δ=0.001** — the tighter margin is not met.
- **Gaussian NLLS dominance on T1** — A4 (0.00116) ≈ 15× better than the VQC,
  because T1's generative model is *exactly* the Gaussian A4 fits. This is a
  task-specific artifact, not a general classical-beats-quantum result.

## Still provisional (to be strengthened in T2-v1)

- **Entanglement's causal contribution** — T1-v2 compared one product-state
  circuit (QP) against fixed entangled circuits; a single hand-built product
  circuit is weak evidence. T2-v1 uses **many paired product/entangled
  architectures**.
- **Trainable quantum angles' contribution** — T1-v2's frozen-vs-trained contrast
  used only the QR architecture (frozen 0.02275 vs trained 0.02219, negligible).
  T2-v1 uses **≥20 architectures, paired frozen-vs-trained with identical initial
  angles**.
- **"Architecture search adds value"** — T1-v2 showed searched (0.017) beats a
  single fixed circuit (0.022), but this is the benefit of **best-of-B validation
  selection**, not proof that a clever search algorithm is needed. T2-v1
  decomposes selection gain at **B=1/5/10/25** and against a distribution of fixed
  random architectures.

## Nesting-adequacy audit of the T1-v2 TOST (post hoc)

**Concern:** the T1-v2 primary paired TOST treated the 30 paired differences
(3 splits × 10 seeds) as i.i.d., which ignores possible within-split correlation
of the differences. (T1-v2 did also report a split-blocked bootstrap CI, which is
split-aware, but the confirmatory TOST itself was i.i.d.)

**New sensitivity check (this package, post hoc):** re-run TOST on **per-split
mean differences** (n = 3 split-level clusters — a conservative cluster-robust /
hierarchical-lite treatment). Result at δ=0.002:

| Pair | i.i.d.-30 TOST 90% CI | split-mean-3 TOST 90% CI | equivalent (both) |
|---|---|---|:--:|
| Random − Evolutionary | [−0.00041, +0.00124] | [−0.00081, +0.00164] | ✅ |
| Random − Greedy | [+0.00017, +0.00168] | [+0.00078, +0.00107] | ✅ |
| Evolutionary − Greedy | [−0.00017, +0.00119] | [−0.00064, +0.00165] | ✅ |

The cluster analysis widens the CIs (as expected with n=3) but **every pair
remains equivalent within ±0.002**. The T1-v2 equivalence conclusion is therefore
**robust to the nesting treatment** — not an artifact of the i.i.d. assumption.

**Correction applied to claims:** none downgraded. The i.i.d.-TOST limitation is
noted, and the equivalence conclusion survives a split-aware re-analysis. **T2-v1
adopts a hierarchical/split-blocked sensitivity analysis as standard** (not only
i.i.d. TOST), so this limitation does not recur.
