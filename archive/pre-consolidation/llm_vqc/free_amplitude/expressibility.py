"""Descriptive architecture diagnostics: Expressibility (fidelity KL vs.
Haar) and Entanglement Capability (Meyer-Wallach Q).

**These are descriptive diagnostics, not assumed predictors of RMSE.**
Nothing here claims a causal or predictive relationship to task
performance; the reporting layer plots them against validation RMSE but
treats any correlation as observation, not mechanism.

Two flavors (correction pass "Expressibility and Entanglement Capability"):

- **Intrinsic** (`intrinsic_diagnostics`): a property of the ARCHITECTURE
  alone. The searched body is applied to |0...0> (amplitude encoding
  EXCLUDED) at many random `theta ~ Uniform[-pi, pi]^P` draws; the same
  sampled states feed both metrics. Cached by `architecture_hash` because
  they do not depend on any candidate's proposed angles. It is a category
  error to call a single fixed-theta value "expressibility" -- expressibility
  is a distributional property over the whole parameter space, which is
  why `intrinsic_diagnostics` requires many samples and this module exposes
  no "expressibility of one theta" function at all.

- **Task-conditioned** (`task_conditioned_entanglement`): the Meyer-Wallach
  Q *distribution* of ONE complete candidate (fixed proposed theta)
  evaluated on the actual amplitude-encoded inputs -- a per-candidate
  descriptive number, explicitly distinct from the intrinsic metric.

Expressibility definition (Sim et al. 2019): sample pairwise state
fidelities `F = |<psi_i|psi_j>|^2`, histogram them, and compute the KL
divergence to the analytic n-qubit Haar fidelity distribution
`P_Haar(F) = (2^n - 1)(1 - F)^(2^n - 2)`. **Lower KL = more expressible.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.ir.compiler_pennylane import (
    _ENTANGLE_GATES,
    _ENTANGLE_PARAM_GATES,
    _NONPARAM_ROTATION_GATES,
    _ROTATION_GATES,
)
from llm_vqc.ir.expand import EntangleInstruction, RotationInstruction, build_program
from llm_vqc.ir.schema import CircuitIR

DIAGNOSTIC_VERSION = "free_amplitude_diagnostics_v1"
DIAGNOSTIC_SEED_BASE = 20260724
DIAGNOSTIC_STATE_SAMPLES = 200
DIAGNOSTIC_FIDELITY_PAIRS = 2000
DIAGNOSTIC_HISTOGRAM_BINS = 75
THETA_DISTRIBUTION = "Uniform[-pi, pi]"
HAAR_FORMULA = "P(F) = (2^n - 1) * (1 - F)^(2^n - 2)"
#: Laplace-style smoothing added to every sampled-histogram bin before the
#: KL, so an empty bin contributes a finite (not infinite) term.
KL_SMOOTHING_EPSILON = 1e-12


@dataclass
class EntanglementStats:
    mean: float
    median: float
    std: float
    min: float
    max: float


@dataclass
class IntrinsicDiagnostics:
    architecture_hash: str
    n_qubits: int
    quantum_parameter_count: int
    expressibility_kl: float
    entanglement_capability: EntanglementStats
    diagnostic_version: str = DIAGNOSTIC_VERSION
    diagnostic_state_samples: int = DIAGNOSTIC_STATE_SAMPLES
    diagnostic_fidelity_pairs: int = DIAGNOSTIC_FIDELITY_PAIRS
    histogram_bins: int = DIAGNOSTIC_HISTOGRAM_BINS
    theta_distribution: str = THETA_DISTRIBUTION
    haar_formula: str = HAAR_FORMULA
    smoothing_rule: str = f"laplace epsilon={KL_SMOOTHING_EPSILON} on sampled histogram"
    diagnostic_seed: int = 0
    fidelity_histogram: list[float] = field(default_factory=list)
    haar_histogram: list[float] = field(default_factory=list)
    bin_edges: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "architecture_hash": self.architecture_hash,
            "n_qubits": self.n_qubits,
            "quantum_parameter_count": self.quantum_parameter_count,
            "expressibility_kl": self.expressibility_kl,
            "expressibility_note": "lower KL = more expressible",
            "entanglement_capability": {
                "mean": self.entanglement_capability.mean,
                "median": self.entanglement_capability.median,
                "std": self.entanglement_capability.std,
                "min": self.entanglement_capability.min,
                "max": self.entanglement_capability.max,
            },
            "diagnostic_version": self.diagnostic_version,
            "diagnostic_state_samples": self.diagnostic_state_samples,
            "diagnostic_fidelity_pairs": self.diagnostic_fidelity_pairs,
            "histogram_bins": self.histogram_bins,
            "theta_distribution": self.theta_distribution,
            "haar_formula": self.haar_formula,
            "smoothing_rule": self.smoothing_rule,
            "diagnostic_seed": self.diagnostic_seed,
        }


def _body_state_qnode(ir: CircuitIR):
    """A QNode that applies ONLY the searched body (no amplitude encoding)
    to |0...0> and returns the statevector."""
    program = build_program(ir)
    dev = qml.device("default.qubit", wires=program.n_qubits)

    def circuit(weights):
        for instr in program.body:
            if isinstance(instr, RotationInstruction):
                if instr.param_index is not None:
                    _ROTATION_GATES[instr.gate](weights[instr.param_index], wires=instr.wire)
                else:
                    _NONPARAM_ROTATION_GATES[instr.gate](wires=instr.wire)
            elif isinstance(instr, EntangleInstruction):
                if instr.param_index is not None:
                    _ENTANGLE_PARAM_GATES[instr.gate](
                        weights[instr.param_index], wires=[instr.control, instr.target]
                    )
                else:
                    _ENTANGLE_GATES[instr.gate](wires=[instr.control, instr.target])
        return qml.state()

    return qml.QNode(circuit, dev), program.num_parameters, program.n_qubits


def _meyer_wallach_q(state: np.ndarray, n_qubits: int) -> float:
    """Q = 2 * (1 - (1/n) * sum_k Tr(rho_k^2)). 0 for product states, 1 for
    maximally entangled single-qubit reductions."""
    tensor = state.reshape([2] * n_qubits)
    purity_sum = 0.0
    for k in range(n_qubits):
        traced_axes = [a for a in range(n_qubits) if a != k]
        rho_k = np.tensordot(tensor, tensor.conj(), axes=(traced_axes, traced_axes))
        purity_sum += float(np.real(np.trace(rho_k @ rho_k)))
    return float(2.0 * (1.0 - purity_sum / n_qubits))


def _haar_bin_probabilities(bin_edges: np.ndarray, n_qubits: int) -> np.ndarray:
    """Analytic Haar probability mass per fidelity bin, from the CDF
    `1 - (1 - F)^(N-1)` (N = 2^n_qubits)."""
    n_states = 2**n_qubits
    lower = bin_edges[:-1]
    upper = bin_edges[1:]
    return (1.0 - lower) ** (n_states - 1) - (1.0 - upper) ** (n_states - 1)


def intrinsic_diagnostics(ir: CircuitIR, architecture_hash: str) -> IntrinsicDiagnostics:
    """Compute intrinsic Expressibility (KL vs. Haar) and Entanglement
    Capability (Meyer-Wallach Q distribution) for an architecture.
    Deterministic given `architecture_hash` (versioned seed)."""
    qnode, n_params, n_qubits = _body_state_qnode(ir)
    seed = derive_child_seed(DIAGNOSTIC_SEED_BASE, "intrinsic", architecture_hash)
    rng = np.random.default_rng(seed)

    thetas = rng.uniform(-np.pi, np.pi, size=(DIAGNOSTIC_STATE_SAMPLES, max(n_params, 1)))
    states = np.array(
        [
            np.asarray(qnode(thetas[i, :n_params]), dtype=complex)
            for i in range(DIAGNOSTIC_STATE_SAMPLES)
        ]
    )

    # Entanglement capability: MW Q over the SAME sampled states.
    q_values = np.array([_meyer_wallach_q(states[i], n_qubits) for i in range(len(states))])
    ent = EntanglementStats(
        mean=float(np.mean(q_values)), median=float(np.median(q_values)),
        std=float(np.std(q_values)), min=float(np.min(q_values)), max=float(np.max(q_values)),
    )

    # Expressibility: pairwise fidelities over random pairs of sampled states.
    idx_a = rng.integers(0, len(states), size=DIAGNOSTIC_FIDELITY_PAIRS)
    idx_b = rng.integers(0, len(states), size=DIAGNOSTIC_FIDELITY_PAIRS)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % len(states)  # avoid trivial self-pairs (F == 1)
    overlaps = np.einsum("ij,ij->i", states[idx_a].conj(), states[idx_b])
    fidelities = np.abs(overlaps) ** 2

    bin_edges = np.linspace(0.0, 1.0, DIAGNOSTIC_HISTOGRAM_BINS + 1)
    sampled_counts, _ = np.histogram(fidelities, bins=bin_edges)
    sampled_prob = sampled_counts.astype(np.float64) + KL_SMOOTHING_EPSILON
    sampled_prob /= sampled_prob.sum()
    haar_prob = _haar_bin_probabilities(bin_edges, n_qubits)
    haar_prob = np.clip(haar_prob, KL_SMOOTHING_EPSILON, None)
    haar_prob /= haar_prob.sum()

    kl = float(np.sum(sampled_prob * np.log(sampled_prob / haar_prob)))

    return IntrinsicDiagnostics(
        architecture_hash=architecture_hash, n_qubits=n_qubits,
        quantum_parameter_count=n_params, expressibility_kl=kl, entanglement_capability=ent,
        diagnostic_seed=int(seed), fidelity_histogram=sampled_prob.tolist(),
        haar_histogram=haar_prob.tolist(), bin_edges=bin_edges.tolist(),
    )


class IntrinsicDiagnosticsCache:
    """In-memory cache keyed by `architecture_hash` (correction pass:
    "Cache these values by architecture_hash because they do not depend on
    the candidate's proposed theta")."""

    def __init__(self) -> None:
        self._cache: dict[str, IntrinsicDiagnostics] = {}

    def get_or_compute(self, ir: CircuitIR, architecture_hash: str) -> IntrinsicDiagnostics:
        if architecture_hash not in self._cache:
            self._cache[architecture_hash] = intrinsic_diagnostics(ir, architecture_hash)
        return self._cache[architecture_hash]

    def __contains__(self, architecture_hash: str) -> bool:
        return architecture_hash in self._cache

    def __len__(self) -> int:
        return len(self._cache)


def task_conditioned_entanglement(
    ir: CircuitIR, theta: list[float], encoded_inputs: np.ndarray
) -> EntanglementStats:
    """Meyer-Wallach Q distribution of ONE complete candidate (fixed
    proposed `theta`) over the ACTUAL amplitude-encoded inputs. Distinct
    from intrinsic diagnostics -- this uses the real data and the real
    proposed angles, and is a per-candidate descriptive number."""
    from llm_vqc.ir.compiler_pennylane import apply_program_ops

    program = build_program(ir)
    dev = qml.device("default.qubit", wires=program.n_qubits)

    def circuit(inputs, weights):
        apply_program_ops(program, inputs, weights)
        return qml.state()

    qnode = qml.QNode(circuit, dev)
    q_values = []
    for i in range(len(encoded_inputs)):
        state = np.asarray(qnode(encoded_inputs[i], theta), dtype=complex)
        q_values.append(_meyer_wallach_q(state, program.n_qubits))
    q = np.array(q_values)
    return EntanglementStats(
        mean=float(np.mean(q)), median=float(np.median(q)), std=float(np.std(q)),
        min=float(np.min(q)), max=float(np.max(q)),
    )
