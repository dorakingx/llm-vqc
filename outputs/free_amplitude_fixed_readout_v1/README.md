# free_amplitude_fixed_readout_v1

> **MOCK/NON-LLM AMPLITUDE-ENCODING SMOKE RUN**

Free-form parameterized-gate VQC search with amplitude encoding and a
fixed quantum readout. The model has NO classical trainable layer: amplitude
encoding in, a free 1-5 gate quantum body, `mu_hat = (1 - <Z_0>) / 2`
out. `classical_parameter_count` is always exactly 0.

## Configuration
- n_qubits: 3. feature_count: 8 (= 2^3).
- readout_qubit: 0 (fixed; never selected by the LLM, never trained).
- max_gates: 5.
- dataset_profile: amplitude_n3_smoke_v1.
- Search arms: Random, LLM Open-loop, LLM Closed-loop -- duplicate detection is
  arm-local (each arm has its own cache namespace).

## Files
- `report.md` -- baselines, every candidate, selected architecture per arm.
- `EXPERIMENT_CONFIG.json`, `dataset_summary.json`, `search_space_summary.json`.
- `candidate_trace.csv`, `training_summary.csv`, `architecture_diagnostics.csv`,
  `pareto_summary.csv`, `selected_circuits.json`.
- `figures/validation_rmse_by_arm.{png,svg}`, `figures/pareto_depth_vs_rmse.{png,svg}`.

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here.
