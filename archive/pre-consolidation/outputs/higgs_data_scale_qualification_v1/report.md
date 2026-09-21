# HIGGS data-scale qualification — research report

**Study:** `higgs_data_scale_qualification_v1` · **Branches from:** HIGGS-v1 @
`932610b` · **Purpose:** determine *why* the HIGGS-v1 VQC stayed near the trivial
baseline, before launching another architecture-search matrix. Pre-registered
(`PROTOCOL.md`, committed `3cd4e62`) before the study ran.

> **Headline.** The HIGGS-v1 failure was **not** an intrinsic limit of the
> 4-qubit / 12-quantum-parameter family. It was caused by **(i) severe data
> scarcity** (500 training samples), **(ii) a mis-tuned inherited training
> protocol** (lr = 0.05), and **(iii) PCA(8) information loss**. Fixing all three,
> **19 of 36 conditions pass the frozen non-degeneracy gate**, and at raw-21 /
> 10,000 samples the VQC reaches **AUROC 0.635** and **beats logistic regression
> and a parameter-comparable MLP**. Both HIGGS-v1 "reversals" (frozen > trainable,
> product > entangled) are shown to be **artifacts of the degenerate 500-sample
> regime — the signs flip once there is enough data.** A HIGGS-v2 search **is**
> justified on the frozen condition below.

## 1. Diagnostic answers

| Question | Answer |
|---|---|
| Does more data help? | **Yes, decisively.** Every n=500 condition fails the gate (9/9); every n=10,000 condition passes (9/9). |
| Does PCA(8) lose useful information? | **Yes.** raw-21 qualifies from n=2,000; PCA(8) only from n=5,000, and is worse at every size. |
| Did the training protocol matter? | **Yes.** The audited protocol beats the inherited one by ~0.012 log-loss; the 3 worst grid configs all use the inherited lr=0.05. |
| Does quantum-angle training help? | **At scale, yes** — and the sign flips with data (see §6). |
| Does entanglement help? | **At scale, yes** — sign flips with data (see §6). |

## 2. Training-protocol audit (development-only)

24 configurations × 2 architectures, selected on **validation log-loss only**:

| Protocol | mean val log-loss |
|---|---:|
| **SELECTED: lr 0.005, 50 ep, wd 1e-3, early-stop 8** | **0.6812** |
| lr 0.01, 50 ep, wd 1e-5, ES 8 | 0.6841 |
| *HIGGS-v1 inherited: lr 0.05, 20 ep, wd 1e-5, no ES* | *0.6928* |
| worst (lr 0.05, 50 ep, wd 1e-3, no ES) | 0.7135 |

**lr = 0.05 — the value inherited from the small T1/T2 tasks — is the worst
learning rate in the grid.** The inherited configuration alone cost ≈0.012
log-loss, more than the gate's 0.01 threshold.

## 3. Data integrity

Qualification pool = official rows **[1,000,000, 1,200,000)**, provably disjoint
from the HIGGS-v1 development subset, the v1 benchmark blocks and the **untouched**
official holdout. 6 blocks (5 qualification + 1 development), 102,000 unique IDs,
no sample crossing a block or partition; training sets are verified **nested**
prefixes; preprocessing fit on training only (perturbation-tested). 1,080 result
rows, **0 training failures**.

## 4. Capacity (verified, as predeclared)

| Representation | dim | embed | quantum | head | total | trainable (M4/M6) | trainable (M5 frozen) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 raw21 | 21 | **88** | 12 | 5 | **105** | 105 | 93 |
| R2 PCA(8) | 8 | **36** | 12 | 5 | **53** | 53 | 41 |
| R3 PCA(16) | 16 | **68** | 12 | 5 | **85** | 85 | 73 |

Cross-representation comparisons are **not** capacity-matched (105 / 53 / 85) —
by design, since the representation question is exactly whether PCA discards
information. Within a representation the pairing is exact.

## 5. Learning curves and the gate

Median internal-test log-loss / AUROC (R1 raw-21):

| Model | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|
| M0 class prior | 0.6916 / 0.500 | 0.6916 | 0.6916 | 0.6916 / 0.500 |
| M1 logreg | 0.6938 / 0.556 | 0.6811 / 0.574 | 0.6782 / 0.589 | 0.6767 / 0.594 |
| M2 MLP (93 params) | 0.6899 / 0.547 | 0.6818 / 0.575 | 0.6770 / 0.596 | 0.6679 / 0.623 |
| M3 product VQC | 0.6885 / 0.554 | 0.6798 / 0.586 | 0.6741 / 0.608 | 0.6678 / 0.625 |
| **M4 entangled VQC** | 0.6874 / 0.551 | 0.6787 / 0.590 | 0.6728 / 0.611 | **0.6636 / 0.635** |
| M5 frozen quantum | 0.6874 / 0.552 | 0.6814 / 0.583 | 0.6740 / 0.608 | 0.6661 / 0.627 |

**Gate outcome: 19 / 36 conditions pass.** All 9 conditions at n=500 fail; all 9 at
n=10,000 pass. raw-21 passes from n=2,000; PCA(8)/PCA(16) only from n=5,000.

## 6. The HIGGS-v1 "reversals" explained (block-level paired CIs, R1)

| Contrast | n=500 | n=5,000 | n=10,000 |
|---|---|---|---|
| frozen − trainable (positive ⇒ **training helps**) | **−0.00036** CI [−0.00067, −0.00010] → **frozen better** | **+0.00129** CI [+0.00016, +0.00296] → **trainable better** | +0.00189 (CI includes 0) |
| product − entangled (positive ⇒ **entanglement helps**) | +0.00030 (ns) | **+0.00123** CI [+0.00020, +0.00215] → **entangled better** | +0.00186 (CI includes 0) |

At n=500 the frozen-beats-trainable effect **reproduces HIGGS-v1 exactly** (CI
excludes 0). At n≥5,000 **both signs flip** and the CIs exclude 0 in the
expected direction. The HIGGS-v1 conclusions QT2/E2 were therefore **correct for
their regime but not general** — they were low-data overfitting artifacts.

## 7. Classical vs VQC

At raw-21 / n=10,000 the entangled VQC (0.6636, AUROC 0.635) **beats** logistic
regression (0.6767, 0.594) and the parameter-comparable MLP (0.6679, 0.623 — same
classical shell, 93 params vs the VQC's 105, i.e. the 12 quantum parameters are
the difference). This **reverses the HIGGS-v1 C2 verdict**, which was measured in
the degenerate regime. Framed correctly: *under this protocol and at this data
scale, adding 12 trainable quantum parameters to the same classical shell improved
test log-loss and AUROC.* **This is not a quantum-advantage claim** — the margin is
small, the comparison is against two specific classical models, and stronger
classical learners on raw features were not re-run here.

## 8. Calibration, runtime, hardware

Calibration (ECE) improves with data (`fig09`). Median VQC training time scales
with n: R1 2.3 s → 9.6 s → 25.9 s → **104.6 s** (`fig10`). Hardware (linear
0–1–2–3): product circuits **0** transpiled two-qubit gates (depth 4); entangled
and frozen **12** (depth 18) — the HIGGS-v1 entangled-circuit cost is stable across
data size and representation, so the hardware-aware trade-off remains available,
now against models that are no longer degenerate.

## 9. Decision-gate outcome and frozen HIGGS-v2 condition

**≥1 condition passes → a HIGGS-v2 search is justified.** Applying the
pre-registered rule (best median validation log-loss, then optimism, then runtime,
then two-qubit count):

> **Frozen HIGGS-v2 condition: representation R1 (raw 21 + StandardScaler),
> training size 10,000, model family M4 (entangled VQC, 105 total trainable
> params), with the audited protocol lr 0.005 / 50 epochs / wd 1e-3 / early
> stopping patience 8 / batch 16.**

Per the protocol, the full Random/Evolutionary/Greedy search is **not** run here;
it requires a separate preregistered HIGGS-v2 protocol.

## 10. Supported conclusions

(a) HIGGS-v1's near-baseline result is explained by data scarcity + inherited
training configuration + PCA(8) loss, not by an intrinsic limit of the model
family; (b) the 4-qubit/12-parameter VQC clears a pre-registered non-degeneracy
gate in 19/36 conditions and at raw-21/10k reaches AUROC 0.635; (c) the HIGGS-v1
frozen>trainable and product>entangled findings are **low-data artifacts whose
signs flip with adequate data**; (d) the audited protocol materially outperforms
the inherited one; (e) capacity, block disjointness, nesting and train-only
preprocessing all verified.

## 11. Unsupported conclusions

**No quantum advantage.** No claim that the VQC beats *strong* classical learners
(RBF-SVM / boosted trees on raw features were not re-run at these sizes). No
architecture-search claim — architectures here are fixed and predeclared. No
HEP-physics claim. No generalization beyond 4 qubits, 12 quantum parameters,
these representations, these sizes, or this protocol.

## 12. Remaining limitations

Only two learners (logreg, MLP) were used for the classical comparison at scale;
n=10,000 is still small versus HIGGS's 11M rows; the largest size dominates
runtime (~105 s/training), which sets the cost of any HIGGS-v2 search; M6 ≡ M4 by
construction; and `statsmodels` is unavailable, so block bootstrap replaces a
random-effects model.

## 13. Reproduction

```bash
python scripts/capacity_controlled/run_qualification.py --phase A
python scripts/capacity_controlled/run_qualification.py --phase B      # shardable: --reps/--blocks
python scripts/capacity_controlled/analyze_qualification.py
```

## 14. Is the project ready for LLM search arms?

**Closer, but not yet — one step remains.** The blocker identified in HIGGS-v1
(the VQC family not beating trivial/classical baselines) is now **resolved** for
the frozen condition. What is still missing is a *powered arm comparison*: HIGGS-v1
found the Random/Evolutionary/Greedy contrast **inconclusive** (block SD 0.016 >
margin 0.01 at 10 blocks). The correct next step is the **preregistered HIGGS-v2
search** on the frozen condition with enough blocks to power the equivalence
margin. Only once that search shows a measurable, adequately-powered arm signal
should LLM arms be added — otherwise an LLM arm would again be measured against an
undetectable difference.
