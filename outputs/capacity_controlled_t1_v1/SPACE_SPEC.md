# Capacity-controlled search space — `T1_capacity_controlled_v1`

This specification is defined by the code in
`llm_vqc/experiments/capacity_controlled/space.py` and was fixed **before** any
performance result was inspected. Its purpose: make **model capacity identical**
for every candidate so that only the internal VQC architecture varies.

## Fixed (identical for every candidate and every arm)

| Property | Value | Consequence |
|---|---|---|
| Task | T1 Gaussian-peak regression, data-split seed 0 | shared frozen split |
| Qubits | **4** | fixed |
| Encoding | angle, gate **RY**, wires **all 4** | encoding width = 4 → embed `Linear(21→4)` |
| Amplitude encoding | **forbidden** | no `2**n` blow-up of classical embed |
| Measurement | **Z** on **all 4** wires | 4 measurements → head `Linear(4→1)` |
| Classical embed | `Linear(21→4)` | **88** params, fixed |
| Classical head | `Linear(4→1)` | **5** params, fixed |
| Output | sigmoid | fixed |
| Trainable **quantum** params | **exactly 12** | fixed |
| **Total trainable params** | **exactly 105** (88 + 12 + 5) | fixed for every candidate |
| Training | AdamW, lr 0.05, 20 epochs, batch 16, wd 1e-5, LR×0.5 @ {7,13,17}, CPU, float64, no early stopping | shared |
| Init policy | `explicit_kaiming_uniform_qc_v1` (see below) | shared, never per-arm |

## Varies (the only scientific degree of freedom)

Internal VQC architecture: parameterized gate **types** (RX/RY/RZ single-qubit
rotations vs CRZ ring), fixed gate **types** (H, CNOT, CZ), gate **ordering**,
entanglement **connectivity** (ring/line/star via CNOT/CZ), and layer
organization.

## Why exactly 12 quantum parameters

A parameterized block spans all 4 wires and contributes exactly
`PARAM_BLOCK_SIZE = 4` angles — either a single-qubit rotation layer (one of
RX/RY/RZ on every wire) or a CRZ ring (4 edges). The count must therefore be a
multiple of 4. Rationale for the *smallest defensible* value:

- **4** (1 block): cannot mix a rotation layer with any CRZ entangler.
- **8** (2 blocks): only two parameterized blocks — limited mixing.
- **12** (3 blocks): **smallest** count that lets a candidate hold multiple
  single-qubit rotation layers *and* multiple CRZ entangling layers at once, i.e.
  real entanglement variation among parameterized gates, while staying
  computationally light (4-qubit sim). Verified to admit **2,139 distinct
  architectures in 3,000 random draws**, and to be sampled/mutated/crossed with
  **zero** rejections (exact count preserved by construction).

12 was chosen on these structural grounds alone, before any RMSE was seen.

## Structural limits (identical across arms)

| Limit | Value |
|---|---|
| max blocks | 6 (3 parameterized + ≤3 free) |
| max depth | 30 |
| max total gates | 32 |
| max two-qubit gates | 24 |

Generator-produced circuits observed within these limits (depth ≤ 25, gates ≤ 28,
two-qubit ≤ 24); the limits still reject pathological hand-built circuits.

## Block vocabulary

- **Parameterized blocks** (4 angles each): `RX`, `RY`, `RZ` (rotation on all
  wires) or `CRZ` (ring). Exactly 3 per candidate.
- **Free blocks** (0 angles): CNOT/CZ in ring/line/star, or an H layer. 0–3 per
  candidate; supply entanglement/connectivity variation at fixed parameter count.

## Explicit initialization policy — `explicit_kaiming_uniform_qc_v1`

Declared before results (never tuned per arm, never chosen post hoc):

- classical embed/head weights **and** biases: `U[-1/√fan_in, +1/√fan_in]`
  (PyTorch `Linear` semantics, made explicit);
- quantum angles: `U(-π, +π)` (full rotation-angle range, symmetric about 0);
- all draws from one `torch.Generator` seeded by the run's `param_init`
  sub-seed, fixed order `embed → quantum → head`; diff method `best` (backprop on
  `default.qubit`).

Legacy runs remain labeled as dependency-default initialization; this policy is
never retro-applied to them.
