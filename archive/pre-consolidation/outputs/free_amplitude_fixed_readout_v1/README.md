# free_amplitude_fixed_readout_v1 (joint structure-and-theta search)

> **REAL-LLM JOINT-SEARCH RUN (OpenAI)**

Main mode: the LLM/sampler proposes a COMPLETE candidate (structure + numerical
theta); the circuit is evaluated at exactly those angles with NO optimizer and
NO classical layer. Prediction: `mu_hat = (1 - <Z_0>) / 2`.
`classical_parameter_count` is always 0.

## Configuration (derived from the dataset profile)
- dataset_profile: amplitude_n3_smoke_v1
- n_qubits: 3, feature_count: 8 (= 2^3)
- readout_qubit: 0 (fixed). max_gates: 5.
- budget per arm: 4. seeds: 2.
- Arms: Random, LLM Open-loop, LLM Closed-loop. Duplicate detection is by
  `candidate_hash` (structure + theta) and is arm-and-seed-local.

## Identity hashes
- `architecture_hash`: gates / order / wires only (keys the cached
  expressibility/entanglement diagnostics).
- `candidate_hash`: architecture + canonical theta (a different theta is a
  different candidate).

## Files
- `report.md`, `EXPERIMENT_CONFIG.json`, `dataset_summary.json`,
  `search_space_summary.json`, `cross_seed_summary.json`.
- `candidate_trace.csv`, `selected_circuits.json`, `architecture_diagnostics.csv`.
- `figures/` -- 14 aggregate figures (PNG + SVG + source CSV each) plus
  `figures/selected_circuits/` diagrams.

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here.
