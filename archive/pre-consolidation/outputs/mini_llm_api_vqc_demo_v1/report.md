# Mini LLM-VQC API Demo -- Report

**run_status: corrected**

This is an end-to-end integration demonstration with n=1 seed.
It does not establish that any search arm is superior.

Provider: OpenAI (`gpt-5.4-mini`). Real API calls made: 4/4. Elapsed: 7.6s.

Duplicate detection and closed-loop feedback are arm-local; closed-loop
feedback always includes validation RMSE, depth, and two-qubit-gate count
(or explicit null) for the arm's own previous proposal.

## Every candidate

| arm | idx | layer1 | entangler | dir | layer2 | valid | dup | trained | train MSE | val RMSE | depth | gates | 2q | angles_changed | dtype_ok | latency(s) | tokens(in/out) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0 | RZ | CZ | 0_to_1 | RY | True | False | True | 0.0033 | 0.0499 | 4 | 7 | 1 | True | True | n/a | n/a |
| Random | 1 | RX | CNOT | 0_to_1 | RX | True | False | True | 0.0017 | 0.0403 | 4 | 7 | 1 | True | True | n/a | n/a |
| LLM Open-loop | 0 | RY | CNOT | 0_to_1 | RY | True | False | True | 0.0091 | 0.0884 | 4 | 7 | 1 | True | True | 2.54 | 412/38 |
| LLM Open-loop | 1 | RY | CNOT | 0_to_1 | RY | True | True | False | 0.0091 | 0.0884 | 4 | 7 | 1 | True | True | 1.30 | 412/38 |
| LLM Closed-loop | 0 | RY | CNOT | 0_to_1 | RY | True | False | True | 0.0091 | 0.0884 | 4 | 7 | 1 | True | True | 0.91 | 418/38 |
| LLM Closed-loop | 1 | RY | CZ | 0_to_1 | RX | True | False | True | 0.0060 | 0.0680 | 4 | 7 | 1 | True | True | 1.42 | 518/37 |

## Selected architecture per arm

### Random
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 2, 'num_invalid': 0, 'num_duplicate': 0, 'num_failed': 0, 'num_unique': 2, 'consumed_budget': 2}
- API: 0 ok / 0 failed / 0 outbound attempts
- selected architecture: `{'layer_1_gate': 'RX', 'entangler': 'CNOT', 'entangler_direction': '0_to_1', 'layer_2_gate': 'RX'}`
- structural hash: `daed189af4fcc545c7013b33f4397d94f68bc0463efe0125686208d167072e60`
- selected validation RMSE: 0.0403
- protected-test RMSE: 0.0391
- total trainable parameters: 51

### LLM Open-loop
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 1, 'num_invalid': 0, 'num_duplicate': 1, 'num_failed': 0, 'num_unique': 1, 'consumed_budget': 2}
- API: 2 ok / 0 failed / 2 outbound attempts
- selected architecture: `{'layer_1_gate': 'RY', 'entangler': 'CNOT', 'entangler_direction': '0_to_1', 'layer_2_gate': 'RY'}`
- structural hash: `ab736ece2bc31773e8a83fc3c93be3e24fae483be0e800b2c03cae81da7d3274`
- selected validation RMSE: 0.0884
- protected-test RMSE: 0.0942
- total trainable parameters: 51

### LLM Closed-loop
- stop reason: budget exhausted normally
- ledger: {'num_proposed': 2, 'num_valid': 2, 'num_invalid': 0, 'num_duplicate': 0, 'num_failed': 0, 'num_unique': 2, 'consumed_budget': 2}
- API: 2 ok / 0 failed / 2 outbound attempts
- selected architecture: `{'layer_1_gate': 'RY', 'entangler': 'CZ', 'entangler_direction': '0_to_1', 'layer_2_gate': 'RX'}`
- structural hash: `ca567c70e8601dfca898e4157ecdd133c08d00eec606377426f71688cf1d20d4`
- selected validation RMSE: 0.0680
- protected-test RMSE: 0.0731
- total trainable parameters: 51

(Initial and learned quantum angles, and classical-weight summaries, are in `selected_circuits.json` and `LEARNED_PARAMETERS.md`.)

---
This is an end-to-end integration demonstration with n=1 seed.
It does not establish that any search arm is superior.