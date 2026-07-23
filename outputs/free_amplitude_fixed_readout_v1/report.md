# MOCK/NON-LLM AMPLITUDE-ENCODING SMOKE RUN

# Free-Amplitude Fixed-Readout Report

This is an integration/smoke demonstration. It does not establish that any search arm is superior, and is not a scientific performance claim.

Elapsed: 1.3s.

## Baselines

| name | trainable params | val RMSE | val MAE | test RMSE | test MAE |
|---|---|---|---|---|---|
| quantum_no_body | 0 | 0.2005 | 0.1775 | 0.2078 | 0.1835 |
| fixed_random_quantum_body | 0 | 0.2005 | 0.1775 | 0.2078 | 0.1835 |

## Every candidate

| arm | idx | valid | dup | val RMSE | depth | 2q gates | q params | params in cone | zero-grad params |
|---|---|---|---|---|---|---|---|---|---|
| Random | 0 | True | False | 0.2818 | 5 | 3 | 4 | 3 | 4 |
| Random | 1 | True | False | 0.2005 | 2 | 1 | 1 | 1 | 1 |
| LLM Open-loop | 0 | True | False | 0.0991 | 4 | 1 | 4 | 3 | 2 |
| LLM Open-loop | 1 | True | True | 0.0991 | 4 | 1 | 4 | 3 | 2 |
| LLM Closed-loop | 0 | True | False | 0.2005 | 3 | 1 | 2 | 0 | 2 |
| LLM Closed-loop | 1 | True | False | 0.2005 | 5 | 2 | 3 | 0 | 3 |

## Selected architecture per arm

### Random
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 2, 'num_invalid': 0, 'num_duplicate': 0, 'num_failed': 0, 'num_unique': 2, 'consumed_budget': 2}
- API: 0 ok / 0 failed / 0 outbound attempts
- selected operations: `[{'gate': 'CRZ', 'wires': [1, 0]}]`
- structural hash: `8fcb19e1854636da8c1be406e99bf0f87057714cf1381c333bcaf5fc31f569b2`
- selected validation RMSE: 0.2005
- protected-test RMSE: 0.2078 (MAE 0.1835)
- classical trainable parameters: 0

### LLM Open-loop
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 1, 'num_invalid': 0, 'num_duplicate': 1, 'num_failed': 0, 'num_unique': 1, 'consumed_budget': 2}
- API: 2 ok / 0 failed / 2 outbound attempts
- selected operations: `[{'gate': 'CRX', 'wires': [1, 0]}, {'gate': 'RX', 'wires': [0]}, {'gate': 'RZ', 'wires': [0]}, {'gate': 'H', 'wires': [2]}, {'gate': 'RY', 'wires': [2]}]`
- structural hash: `67abc3cffbe56f0c9c7861ad18bb3f7a1a4b0be3abbad9962ce9cbb0ff237032`
- selected validation RMSE: 0.0991
- protected-test RMSE: 0.1035 (MAE 0.0900)
- classical trainable parameters: 0

### LLM Closed-loop
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 2, 'num_invalid': 0, 'num_duplicate': 0, 'num_failed': 0, 'num_unique': 2, 'consumed_budget': 2}
- API: 2 ok / 0 failed / 2 outbound attempts
- selected operations: `[{'gate': 'RZ', 'wires': [1]}, {'gate': 'CRY', 'wires': [1, 2]}, {'gate': 'H', 'wires': [2]}, {'gate': 'H', 'wires': [1]}, {'gate': 'CRY', 'wires': [1, 2]}]`
- structural hash: `4e32834701984139155b712e9c10ea310ccaff5b4513190a45dfa0f11e3be199`
- selected validation RMSE: 0.2005
- protected-test RMSE: 0.2078 (MAE 0.1835)
- classical trainable parameters: 0

(Initial/learned quantum angles and full diagnostics are in `selected_circuits.json`, `architecture_diagnostics.csv`, and `candidate_trace.csv`.)

## Amplitude-encoding limitation
L2 amplitude normalization removes overall multiplicative scale: two samples differing only by a positive amplitude factor A become the same normalized quantum state before noise. This model cannot recover absolute signal magnitude; only the peak location (mu) is targeted.

---
# MOCK/NON-LLM AMPLITUDE-ENCODING SMOKE RUN
This is an integration/smoke demonstration, not a scientific performance claim.