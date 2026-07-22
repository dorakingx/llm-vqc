# Model specification — HIGGS data-scale qualification

Seven model families per (block, training size, representation). VQC families use
**5 predeclared architecture seeds** (5000–5004). This is qualification, **not**
architecture search: architectures are fixed and predeclared, never selected on
performance.

| ID | Model | Trainable parameters (by representation) |
|---|---|---|
| **M0** | class-prior baseline (predicts training positive rate) | 0 |
| **M1** | logistic regression (grid C ∈ {0.1, 1, 10}, validation-selected) | dim + 1 |
| **M2** | small classical MLP `dim→4→1`, tanh, sigmoid out | R1 93, R2 41, R3 73 |
| **M3** | fixed **product-state** VQC (no two-qubit gates), trainable angles | embed + 12 + 5 |
| **M4** | fixed **entangled** VQC (≥1 CRZ), trainable angles | embed + 12 + 5 |
| **M5** | **frozen** quantum, exact M4 architecture (only embed+head train) | embed + 5 (12 frozen) |
| **M6** | **trainable** version of the exact M4 architecture | ≡ M4 |

**M6 ≡ M4.** Both denote the trainable M4 architecture, so it is computed once and
reported under both labels (documented, avoids duplicate compute and a spurious
"two families" impression).

## Shared VQC structure (inherited controlled space)

4 qubits, angle-RY on all wires (no amplitude), Z on all wires, **exactly 12
trainable quantum parameters**, classical embed `Linear(dim→4)`, head
`Linear(4→1)`, sigmoid output. The **only** change across representations is the
embed input dimension.

## Capacity accounting (recorded per condition)

| Representation | input dim | embed params | quantum | head | **total params** | **total trainable** (M4/M6) | **total trainable** (M5 frozen) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 raw21 | 21 | **88** | 12 | 5 | **105** | 105 | 93 |
| R2 PCA(8) | 8 | **36** | 12 | 5 | **53** | 53 | 41 |
| R3 PCA(16) | 16 | **68** | 12 | 5 | **85** | 85 | 73 |

Within a representation, M3/M4/M5/M6 preserve the intended pairing (M3, M4 and M6
have identical counts; M5 has the same *total* params with 12 **frozen**).
**Cross-representation comparisons always report the parameter difference** — R1
(105) vs R2 (53) vs R3 (85) are *not* capacity-matched, by design, because the
representation question is precisely whether PCA(8) discards useful information.

## Product ↔ entangled pairing

M3 (product) is obtained from the same architecture family with the exact
`CRZ(θ) → RZ(θ)`-on-target mapping used in HIGGS-v1: one parameter → one
parameter, gate and layer counts preserved, two-qubit count → 0. Product circuits
are verified to contain **no entangling gates**.

## Frozen ↔ trainable pairing

M5 and M4/M6 use the **exact same IR and the exact same initial quantum angles**
(same train seed → same `param_init`), the same classical initialization, the same
minibatch ordering and the same preprocessing; **only `requires_grad` on the
quantum weights differs**. Verified in tests.
