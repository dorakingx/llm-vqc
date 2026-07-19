# HIGGS-v1 capacity-controlled space

> Circuit grammar inherited unchanged from `T1_capacity_controlled_v1`
> (`llm_vqc/experiments/capacity_controlled/space.py`). Only the task/data and the
> embed input dimension differ.

## Fixed for every primary HIGGS candidate

| Property | Value |
|---|---|
| Task | HIGGS 21 low-level → PCA(8), block-train-only preprocessing |
| Qubits | 4 |
| Encoding | angle, gate RY, all 4 wires (**amplitude forbidden**) |
| Measurement | Z on all 4 wires |
| Trainable quantum params | **12** |
| Classical embed | `Linear(8→4)` = **36** params |
| Classical head | `Linear(4→1)` = **5** params |
| **Total trainable params** | **53** = 36 + 12 + 5 (verified per candidate) |
| Structural limits | MAX_BLOCKS=6, MAX_DEPTH=30, MAX_TOTAL_GATES=32, MAX_TWO_QUBIT_GATES=24 |
| Init | `explicit_kaiming_uniform_qc_v1` (quantum angles U(−π,π)) |
| Training | AdamW lr=0.05, 20 epochs, batch 16, wd 1e-5, LR×0.5 @ {7,13,17}, CPU float64, no early stopping |
| Loss / selection | BCE / validation **log-loss** |

## Varies (only)

Internal VQC architecture: parameterized gate types (RX/RY/RZ / CRZ), fixed gate
types (H/CNOT/CZ), ordering, entanglement connectivity, layer organization. Every
candidate keeps exactly 12 quantum angles and 53 total trainable params by
construction; `validate_controlled` is the independent second gate; the analysis
fails loudly if any candidate ≠ 53.

## Capacity progression across tasks

| Task | raw dim | embed | total trainable params |
|---|---:|---|---:|
| T1 | 21 | 88 | 105 |
| T2 | 10 (PCA) | 44 | 61 |
| **HIGGS** | **8 (PCA)** | **36** | **53** |

Only the embed input dimension changes; the circuit IR space is identical.
