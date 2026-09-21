# mini_llm_api_vqc_demo_v1

A minimal, real-OpenAI-API integration demonstration: an LLM proposes a
2-qubit variational circuit's discrete structure (compact 4-field grammar),
AdamW trains its 51 continuous parameters (4 quantum rotation angles) on the
T1 Gaussian-peak-position regression task, validation RMSE selects the best
circuit per arm, and a protected test partition is scored once per arm.

**run_status:** `corrected`. Integration demonstration only (1 seed,
budget=2 per arm, 5 epochs per candidate, 4
real LLM calls). Does not establish that any search arm is superior.

## Corrected-pass guarantees
- Duplicate detection and closed-loop feedback are **arm-local** (no
  cross-arm contamination; arm execution order cannot change results).
- Closed-loop feedback always carries all six fields (architecture,
  validation RMSE-or-null, validity, duplicate status, depth-or-null,
  two-qubit-gate-count-or-null); never any protected-test field.
- Explicit initialization (`mini_demo_init_v1`: Xavier/zeros/Uniform[-pi,pi]),
  `diff_method="backprop"`, all trainable tensors float64 on CPU.
- Provider built with `max_retries=0`; a positive `LLM_API_BUDGET_USD` and
  an explicit `OPENAI_MODEL` are required before any request.

## Configuration
- Task: T1 Gaussian-peak regression (train=150, val=250, protected test=2000).
- Qubits: 2. Quantum parameters: 4. Total trainable parameters: 51.
- Provider: OpenAI (`gpt-5.4-mini`), real API calls, hard cap 4.
- Seed: 0.

## Reproduction
```bash
export OPENAI_API_KEY=...          # never printed or committed
export OPENAI_MODEL=...            # required; no silent default
export LLM_API_BUDGET_USD=2.00     # required positive hard cap
python scripts/run_mini_llm_vqc_demo.py --provider openai --budget 2 \
    --epochs 5 --seed 0 \
    --output outputs/mini_llm_api_vqc_demo_v1
```

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here -- only parsed proposals, aggregate token/latency numbers, and
compact weight summaries.
