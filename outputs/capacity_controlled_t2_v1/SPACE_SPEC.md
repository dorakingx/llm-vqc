# T2-v1 capacity-controlled space

> Circuit grammar **inherited unchanged** from `T1_capacity_controlled_v1`
> (`llm_vqc/experiments/capacity_controlled/space.py`). Only the task/data and the
> classical embed input dimension differ. See `PROTOCOL.md`.

## Fixed for every primary T2 candidate

| Property | Value |
|---|---|
| Task | `T2_digits_3_vs_8` (PCA(10), fit on train only) |
| Qubits | 4 |
| Encoding | angle, gate RY, all 4 wires (**amplitude forbidden**) |
| Measurement | Z on all 4 wires |
| Trainable quantum params | **12** (exactly) |
| Classical embed | `Linear(10→4)` = **44** params |
| Classical head | `Linear(4→1)` = **5** params |
| **Total trainable params** | **61** = 44 + 12 + 5 (verified per candidate) |
| Encoding width | 4 · Measurement width | 4 |
| Structural limits | MAX_BLOCKS=6, MAX_DEPTH=30, MAX_TOTAL_GATES=32, MAX_TWO_QUBIT_GATES=24 |
| Init | `explicit_kaiming_uniform_qc_v1` (quantum angles U(−π,π)) |
| Training | AdamW lr=0.05, 20 epochs, batch 16, wd 1e-5, LR×0.5 @ {7,13,17}, CPU float64, no early stopping |
| Loss | BCE | Selection metric | validation **log-loss** (lower better) |

## Varies (only)

Internal VQC architecture: parameterized gate types (RX/RY/RZ / CRZ), fixed gate
types (H/CNOT/CZ), ordering, entanglement connectivity, layer organization —
exactly as in T1-v2. Every candidate keeps exactly 12 quantum angles and 61 total
trainable params by construction; `validate_controlled` (unchanged) is the
independent second gate; the analysis fails loudly if any candidate ≠ 61.

## Difference from T1-v2

Only the embed input dimension: T1 raw dim 21 → embed 88 → total 105; T2 raw dim
10 (PCA) → embed 44 → **total 61**. The circuit IR space is identical.
