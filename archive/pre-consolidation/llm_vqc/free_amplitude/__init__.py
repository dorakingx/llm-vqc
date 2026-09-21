"""Free-form parameterized-gate VQC search with amplitude encoding and a
fixed quantum readout (`free_amplitude_fixed_readout_v1`).

Everything under this package is scoped to this one experiment. Unlike
`llm_vqc.mini_demo` (which reuses `HybridQNNModel`'s classical embed/head),
this experiment's whole point is a model with **no trainable classical
layer at all** -- amplitude encoding in, a free 1-5 gate quantum body, a
fixed Z(q0) readout, `mu_hat = (1 - z0) / 2` out. That does not fit
`llm_vqc.evaluation.harness.evaluate_candidate` (which always constructs a
`HybridQNNModel`), so this package has its own model, training loop, and
harness, while still reusing `llm_vqc.ir` (schema/validators/compilers/
canonicalize/metrics) and `llm_vqc.ir.budget.BudgetLedger` unchanged.

No real or billed LLM API calls are made by anything in this package as
delivered -- only `llm_vqc.free_amplitude.sampler` (Random search) and
mock/scripted providers are exercised.
"""

from __future__ import annotations
