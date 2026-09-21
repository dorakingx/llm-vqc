"""HIGGS paired quantum ablations.

- **Freeze vs train**: 20 predeclared valid controlled architectures.
- **Entangled vs product** with the *exact* CRZ->RZ mapping required by the
  protocol: for every parameterized entangling slot, `CRZ(theta)` on (control,
  target) becomes `RZ(theta)` on the target wire. In this grammar a CRZ param
  block is a CRZ ring over all 4 wires (targets = every wire), so it maps to an
  RZ rotation layer on all wires — **1 parameter -> 1 parameter, 4 gates -> 4
  gates, layer count preserved, two-qubit count -> 0**. Entangled architectures
  are chosen to contain >=1 CRZ block and NO free entangling blocks, so the
  product counterpart has zero entanglement. Backend depth may still differ
  (CRZ-ring serializes on shared wires); that difference is recorded.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary

FREEZE_ARCH_SEEDS = tuple(range(3000, 3020))       # 20 architectures for freeze/train
_ENTANGLE_TEMPLATES = {i for i, t in enumerate(S.FREE_BLOCK_TEMPLATES) if t[0] == "entangle"}


def predeclared_architectures(seeds=FREEZE_ARCH_SEEDS):
    return [S.random_genome(np.random.default_rng(s)) for s in seeds]


def _has_crz(g) -> bool:
    return any(b.role == "param" and b.kind == "CRZ" for b in g.blocks)


def _has_free_entangle(g) -> bool:
    return any(b.role == "free" and b.template_index in _ENTANGLE_TEMPLATES for b in g.blocks)


def product_counterpart(genome):
    """Exact CRZ->RZ product mapping. CRZ param block -> RZ rotation block
    (RZ(theta) on the target of each CRZ = every wire). Non-CRZ blocks unchanged.
    Preserves param count and layer count; removes two-qubit gates."""
    blocks = []
    for b in genome.blocks:
        if b.role == "param" and b.kind == "CRZ":
            blocks.append(S.ParamBlock(kind="RZ"))
        else:
            blocks.append(b.model_copy())
    return S.ControlledGenome(blocks=blocks)


def entangled_pairs(n=20, start_seed=4000):
    """`n` (entangled, product) pairs. Entangled members have >=1 CRZ block and no
    free entangling block, so their product counterparts are entanglement-free."""
    pairs = []
    s = start_seed
    while len(pairs) < n:
        g = S.random_genome(np.random.default_rng(s)); s += 1
        if _has_crz(g) and not _has_free_entangle(g):
            p = product_counterpart(g)
            if circuit_cost_summary(S.genome_to_ir(p)).two_qubit_gate_count == 0:
                pairs.append((g, p))
    return pairs


def pair_diagnostics(entangled, product) -> dict:
    ce = circuit_cost_summary(S.genome_to_ir(entangled))
    cp = circuit_cost_summary(S.genome_to_ir(product))
    pe = build_program(S.genome_to_ir(entangled)).num_parameters
    pp = build_program(S.genome_to_ir(product)).num_parameters
    return {"entangled_two_qubit": ce.two_qubit_gate_count, "product_two_qubit": cp.two_qubit_gate_count,
            "entangled_gates": ce.gate_count, "product_gates": cp.gate_count,
            "gate_count_match": ce.gate_count == cp.gate_count,
            "entangled_depth": ce.depth, "product_depth": cp.depth, "depth_diff": ce.depth - cp.depth,
            "param_count_preserved": pe == pp == S.QUANTUM_PARAM_COUNT}
