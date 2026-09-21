"""Minimal real-LLM-API VQC architecture-search demonstration (v1).

Everything under this package is deliberately scoped to
`mini_llm_api_vqc_demo_v1` only: a compact 4-field architecture grammar
(NOT the general `llm_vqc.ir.schema.CircuitIR` grammar `llm_vqc.ir.sampler`
and the full-scale search arms use), a matching structured-output OpenAI
provider, and the compact-to-CircuitIR converter. None of this is wired
into the general search framework (`llm_vqc.search`) and it must not be
imported by any T1/T2/HIGGS/qualification experiment code.
"""

from __future__ import annotations
