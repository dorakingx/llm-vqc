"""Structural cost accounting for a free-gate candidate (Codex instruction
sections 7 and 15).

**The searched five-gate budget excludes state preparation.** Amplitude
encoding is one high-level IR operation, but preparing an arbitrary
amplitude state may require a substantial decomposed circuit -- this
module reports both the high-level count (always 1) and a best-effort
*decomposed* gate count/depth via Qiskit transpilation, so "amplitude
encoding is physically depth-one" is never claimed. When decomposition is
unavailable for any reason, the corresponding field is the string
`"unavailable"`, never an invented number.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import CircuitCostSummary, circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR

IntOrUnavailable = int | str


def _generic_amplitude_vector(dim: int) -> list[float]:
    """A deliberately non-trivial amplitude vector (NOT a computational
    basis state) for decomposition-cost probing: Qiskit's `prepare_state`
    recognizes a basis state like `[1,0,0,...]` and emits (near-)zero
    gates for it, which would silently under-report the real cost of
    preparing an arbitrary amplitude state. A linearly-varying, all-nonzero
    vector avoids that degenerate case while staying a fixed, deterministic
    probe (not read from any real sample)."""
    raw = [float(i + 1) for i in range(dim)]
    norm = sum(v * v for v in raw) ** 0.5
    return [v / norm for v in raw]


@dataclass
class StructuralCostSummary:
    n_qubits: int
    logical_encoding_type: str
    high_level_encoding_operation_count: int
    searched_body_gate_count: int
    controlled_gate_count: int
    quantum_parameter_count: int
    searched_body_depth: IntOrUnavailable
    total_compiled_depth: int
    decomposed_state_preparation_gate_count: IntOrUnavailable
    decomposed_state_preparation_depth: IntOrUnavailable
    total_decomposed_gate_count: IntOrUnavailable
    total_decomposed_depth: IntOrUnavailable


def _decomposed_state_prep_cost(ir: CircuitIR) -> tuple[IntOrUnavailable, IntOrUnavailable]:
    try:
        from qiskit import QuantumCircuit, transpile

        program = build_program(ir)
        qc = QuantumCircuit(program.n_qubits)
        dummy_inputs = _generic_amplitude_vector(program.num_inputs)
        qc.prepare_state(dummy_inputs, list(program.amplitude_wires), normalize=True)
        transpiled = transpile(qc, basis_gates=["u3", "cx"], optimization_level=1)
        return transpiled.size(), transpiled.depth()
    except Exception:
        return "unavailable", "unavailable"


def _total_decomposed_cost(ir: CircuitIR) -> tuple[IntOrUnavailable, IntOrUnavailable]:
    try:
        from qiskit import transpile

        from llm_vqc.ir.compiler_qiskit import to_qiskit_bound

        program = build_program(ir)
        dummy_inputs = _generic_amplitude_vector(program.num_inputs)
        dummy_weights = [0.0] * program.num_parameters
        qc = to_qiskit_bound(ir, inputs=dummy_inputs, weights=dummy_weights, program=program)
        transpiled = transpile(qc, basis_gates=["u3", "cx"], optimization_level=1)
        return transpiled.size(), transpiled.depth()
    except Exception:
        return "unavailable", "unavailable"


def _searched_body_depth(ir: CircuitIR) -> IntOrUnavailable:
    """Depth of the searched body ALONE (excluding state preparation) --
    computed by compiling a copy of `ir` with a trivial (angle, single-RY,
    reupload=0 on wire 0) encoding standing in for amplitude, purely to
    isolate the body's own depth via the same Qiskit depth calculation
    `circuit_cost_summary` otherwise uses for the full circuit. Reported as
    `"unavailable"` rather than guessed if this substitution fails."""
    try:
        from llm_vqc.ir.schema import EncodingSpec

        body_only_ir = ir.model_copy(
            update={"encoding": EncodingSpec(type="angle", gate="RY", wires=[0], reupload=0)}
        )
        cost = circuit_cost_summary(body_only_ir)
        return cost.depth
    except Exception:
        return "unavailable"


def structural_cost_summary(ir: CircuitIR) -> StructuralCostSummary:
    program = build_program(ir)
    full_cost: CircuitCostSummary = circuit_cost_summary(ir)

    decomp_prep_gates, decomp_prep_depth = _decomposed_state_prep_cost(ir)
    total_decomp_gates, total_decomp_depth = _total_decomposed_cost(ir)

    return StructuralCostSummary(
        n_qubits=ir.n_qubits,
        logical_encoding_type=ir.encoding.type,
        high_level_encoding_operation_count=1,
        searched_body_gate_count=len(program.body),
        controlled_gate_count=full_cost.two_qubit_gate_count,
        quantum_parameter_count=full_cost.parameter_count,
        searched_body_depth=_searched_body_depth(ir),
        total_compiled_depth=full_cost.depth,
        decomposed_state_preparation_gate_count=decomp_prep_gates,
        decomposed_state_preparation_depth=decomp_prep_depth,
        total_decomposed_gate_count=total_decomp_gates,
        total_decomposed_depth=total_decomp_depth,
    )
