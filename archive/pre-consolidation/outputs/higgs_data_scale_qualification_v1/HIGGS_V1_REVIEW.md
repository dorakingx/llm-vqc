# HIGGS-v1 audit — why the VQC stayed near the trivial baseline

Independently recomputed from the committed HIGGS-v1 machine-readable outputs
(not from its prose). **The HIGGS-v1 package is not modified.**

## Recomputed and confirmed

| Quantity | Recomputed | Status |
|---|---|---|
| Evaluated candidates / unique param counts | 2,005 / {53} | ✅ capacity uniform |
| Search runs / invalid / failed | 90 / 0 / 0 | ✅ |
| VQC protected-test log-loss (median) | Random 0.7287, Evolutionary 0.7139, Greedy 0.7267 | ✅ |
| VQC protected-test AUROC (median) | 0.5183 – 0.5252 | ✅ barely above chance |
| Class-prior baseline (C0 majority) | **0.6917** log-loss | ✅ |
| Strongest classical (RBF-SVM) / fair NAS | 0.6910 / 0.6959 | ✅ |
| External holdout (5,000-sample retrain) | AUROC **0.5722**, log-loss **0.6867** | ✅ |
| Frozen − trainable | −0.0062, CI [−0.0097, −0.0027] (excludes 0) | ✅ frozen better |
| Product − entangled | −0.0087, CI [−0.0114, −0.0061] (excludes 0) | ✅ product better |
| Selection gain (fixed → best-of-10) | 0.7522 → 0.7258 | ✅ |
| Hardware-aware two-qubit / log-loss | 16 → 12 (**−23%**), 0.7261 → 0.7259 | ✅ |

## Carefully worded findings

1. **The 500-sample VQC condition did not beat the trivial class-prior
   baseline.** Median protected-test log-loss was 0.714–0.729 versus **0.6917**
   for the class-prior model, and AUROC was 0.518–0.525 (chance = 0.5). The
   searched VQC was, on this metric, *worse* than predicting the class prior.

2. **The 5,000-sample external retraining improved both AUROC and log-loss.**
   The consensus architecture retrained on the combined 5,000-sample training set
   reached AUROC 0.5722 and log-loss 0.6867 — better than the 500-sample blocks
   on both (AUROC ≈0.52, log-loss ≈0.714) and, unlike them, at least not worse
   than the class prior on log-loss.

3. **This comparison is suggestive but is NOT a controlled learning curve.** The
   two settings differ in more than training-set size: different data pool
   (benchmark blocks vs. the official holdout region), different evaluation set,
   a single consensus architecture rather than the searched distribution, and a
   different preprocessing fit. No sample-size factor was varied while holding
   everything else fixed. It therefore **cannot be read as evidence that data
   scarcity alone caused the failure**.

4. **The current result cannot distinguish the candidate causes.** Data scarcity
   (500 training samples), PCA(8) information loss, training configuration
   inherited from much smaller/easier tasks (lr 0.05, 20 epochs, no early
   stopping), and model capacity (4 qubits, 12 quantum parameters, 53 total) are
   **mutually confounded** in HIGGS-v1. Separating them is precisely the purpose
   of this qualification cycle.

5. **Hardware-aware selection reduced transpiled two-qubit gates by about 25%
   (16 → 12, −23%) with negligible performance loss** (0.7261 → 0.7259 log-loss).
   This is the one clearly positive, reusable HIGGS-v1 result — but note it was
   measured in a regime where the underlying models were near-degenerate, so its
   stability across data size and representation still needs checking.

6. **No LLM experiment is justified yet.** An LLM search arm would be compared
   against a VQC family that does not beat a trivial baseline, and against an
   arm comparison that HIGGS-v1 itself found *inconclusive* (S3, underpowered:
   block SD 0.0159 > margin 0.01, P(equivalence | true 0) = 0.22). Spending LLM
   budget now would measure differences inside a degenerate model family.

## Consequence for this cycle

Before any new Random/Evolutionary/Greedy matrix, run a **factorial qualification
study** that varies training size (500/2,000/5,000/10,000), representation
(raw-21 / PCA(8) / PCA(16)) and model family (class prior, logistic regression,
MLP, product VQC, entangled VQC, frozen-quantum, trainable-quantum) on new,
disjoint qualification blocks, with a development-only training-protocol audit,
and evaluate a **pre-registered non-degeneracy gate**. A full HIGGS-v2 search is
justified **only if** some VQC condition passes that gate.
