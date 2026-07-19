"""Hardware-aware compilation of controlled circuits to a fixed 4-qubit linear
topology 0-1-2-3, with one fixed transpilation configuration and seed.

Records logical vs transpiled depth and two-qubit counts, an inserted-SWAP
estimate, total gates, and trainable parameters, for the hardware-aware secondary
analysis. Deterministic given (ir).
"""

from __future__ import annotations

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.ir.compiler_qiskit import to_qiskit_symbolic
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary

LINEAR_COUPLING = CouplingMap([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
BASIS = ["rz", "ry", "rx", "h", "cx"]
OPT_LEVEL = 1
SEED = 20260719

_TWO_QUBIT = {"cx", "cz", "crz", "swap"}


def _count_2q(qc) -> int:
    return sum(v for k, v in qc.count_ops().items() if k in _TWO_QUBIT)


def transpile_metrics(ir) -> dict:
    qc = to_qiskit_symbolic(ir)
    logical_depth = qc.depth()
    logical_two_qubit = _count_2q(qc)
    # decompose to basis without routing -> logical cx count after translation
    flat = transpile(qc, basis_gates=BASIS, optimization_level=OPT_LEVEL, seed_transpiler=SEED)
    logical_cx = flat.count_ops().get("cx", 0)
    # route onto the linear coupling map
    routed = transpile(qc, coupling_map=LINEAR_COUPLING, basis_gates=BASIS,
                       optimization_level=OPT_LEVEL, seed_transpiler=SEED)
    transpiled_cx = routed.count_ops().get("cx", 0)
    inserted_swaps = max(0, round((transpiled_cx - logical_cx) / 3))  # each SWAP ~ 3 CX
    cost = circuit_cost_summary(ir)
    return {
        "logical_depth": int(logical_depth),
        "transpiled_depth": int(routed.depth()),
        "logical_two_qubit": int(logical_two_qubit),
        "transpiled_two_qubit_cx": int(transpiled_cx),
        "inserted_swaps_est": int(inserted_swaps),
        "total_gates_transpiled": int(sum(routed.count_ops().values())),
        "trainable_parameters": int(build_program(ir).num_parameters + 0),  # quantum params
        "cost_two_qubit_logical": int(cost.two_qubit_gate_count),
    }
