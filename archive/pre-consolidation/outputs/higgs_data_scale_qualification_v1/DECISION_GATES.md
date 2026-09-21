# Decision gates — frozen before the study

## Non-degeneracy gate

A **(representation, training size, VQC model)** condition passes only if **all**
of the following hold on block-level results (5 blocks; architecture and
initialization seeds are repeated measurements *within* a block, never counted as
independent datasets):

| # | Criterion | Threshold |
|---|---|---|
| 1 | median **validation** AUROC | ≥ 0.58 |
| 2 | median **internal-test** AUROC | ≥ 0.58 |
| 3 | median internal-test log-loss improvement over the class-prior baseline | ≥ 0.01 |
| 4 | blocks with positive improvement | ≥ 4 of 5 |
| 5 | training failure rate | < 5% |
| 6 | performance not produced by a single architecture seed | ≥ 2 of the 5 arch seeds individually satisfy criteria 1–3 |
| 7 | validation→test optimism gap (test log-loss − validation log-loss) | < 0.02 |

**This gate is frozen. It will not be weakened after observing outcomes.**

Criterion 6 is operationalized as: at least 2 of the 5 predeclared architecture
seeds independently meet criteria 1–3, so a pass cannot rest on one lucky circuit.

## Outcome routing

### If NO VQC condition passes
- Do **not** run a new architecture-search matrix.
- Conclude: the current **4-qubit, 12-quantum-parameter** family is not a viable
  performance benchmark for HIGGS under the tested conditions.
- Recommend exactly one of: larger quantum-parameter budget; data re-uploading;
  more qubits; a different encoding; or a pivot to hardware-aware
  multi-objective search.

### If ≥1 VQC condition passes
Select the frozen HIGGS-v2 condition by this ordered rule:
1. best median **validation** log-loss;
2. then lowest **validation→test optimism**;
3. then lowest **runtime**;
4. then lowest **transpiled two-qubit gate count**.

Freeze that condition, then create a **separate** preregistered HIGGS-v2 search
protocol. The full Random/Evolutionary/Greedy search is **not** run in this
exploratory stage.

## Supporting (non-gating) determinations

Reported separately, each with block-aware confidence intervals, and **not**
used to modify the gate:

- does more training data help (learning curves);
- does PCA(8) lose useful information (R1 vs R2 vs R3);
- does the audited training protocol help (vs the HIGGS-v1 inherited settings);
- does quantum-angle training help (M5 frozen vs M4/M6 trainable, paired);
- does entanglement help at larger data sizes (M3 product vs M4 entangled, paired);
- is the hardware-aware two-qubit advantage stable across size and representation.

## Readiness for LLM search arms

LLM arms are justified only if a VQC condition passes this gate **and** the
resulting frozen condition clears the class-prior baseline by a margin large
enough that arm differences are measurable. Otherwise the recommendation remains
"not ready".
