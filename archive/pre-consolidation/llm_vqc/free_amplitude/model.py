"""The fixed-readout quantum-only model (Codex instruction sections 2-3):

```text
2^n real-valued input features
        v
L2 normalization (performed once, by the backend AmplitudeEmbedding
        v          normalize=True option -- see module docstring below)
Amplitude Encoding on n qubits
        v
Free searched quantum body with 1-5 gates
        v
Measure Z expectation on fixed readout qubit q0
        v
mu_hat = (1 - <Z0>) / 2
```

**No classical layer of any kind.** There is deliberately no
`torch.nn.Linear` anywhere in this module -- no input embedding, no output
projection, no learned bias, no sigmoid head. `classical_parameter_count`
for this model is always exactly `0`; every trainable parameter is one of
the searched circuit's quantum rotation angles. This is the entire
scientific point of the experiment (Codex instruction section 1): isolate
the effect of circuit structure and quantum-angle training from any
classical layer that could otherwise absorb or launder performance
differences.

**Single normalization location.** `llm_vqc.ir.compiler_pennylane.to_qnode`
already builds `qml.AmplitudeEmbedding(inputs, wires=..., normalize=True,
pad_with=0.0)` for `encoding.type == "amplitude"` circuits -- this module
does NOT additionally normalize the input tensor itself. Normalizing both
here and in the embedding would still be mathematically a no-op (normalizing
an already-unit vector leaves it unchanged) but would violate the "one
normalization location" requirement in spirit (two independent code paths
computing/relying on normalization). The pre-normalization L2 norm is
computed and recorded separately, purely for diagnostics -- see
`llm_vqc.free_amplitude.tasks`.

Because `feature_count` must equal exactly `2 ** n_qubits` (validated
upstream, never padded/truncated here), `pad_with=0.0` in the shared
compiler is inert for this experiment's inputs.
"""

from __future__ import annotations

import pennylane as qml
import torch

from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


class FixedReadoutModelError(Exception):
    """Raised when a CircuitIR does not satisfy this model's fixed contract
    (amplitude encoding, exactly one Z-readout measurement wire)."""


class FixedReadoutQuantumModel(torch.nn.Module):
    """`raw_features (amplitude vector) -> free quantum body -> Z(readout)
    -> mu_hat = (1 - z0) / 2`. No embed, no head, no sigmoid head."""

    def __init__(self, ir: CircuitIR, readout_qubit: int) -> None:
        super().__init__()
        if ir.encoding.type != "amplitude":
            raise FixedReadoutModelError(
                f"FixedReadoutQuantumModel requires amplitude encoding, got {ir.encoding.type!r}"
            )
        if ir.measurements.wires != [readout_qubit]:
            raise FixedReadoutModelError(
                f"FixedReadoutQuantumModel requires measurements.wires == [{readout_qubit}], "
                f"got {ir.measurements.wires!r}"
            )
        if ir.measurements.observable != "Z":
            raise FixedReadoutModelError(
                "FixedReadoutQuantumModel requires observable 'Z', got "
                f"{ir.measurements.observable!r}"
            )

        program: CircuitProgram = build_program(ir)
        self.readout_qubit = readout_qubit
        self.expected_feature_count = program.num_inputs
        self.quantum_parameter_count = program.num_parameters

        qnode = to_qnode(ir, diff_method="backprop")
        weight_shapes = {"weights": (program.num_parameters,)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        z0 = self.q_layer(raw_features)
        mu_hat = (1.0 - z0) / 2.0
        return mu_hat

    @property
    def classical_parameter_count(self) -> int:
        """Always 0 -- no classical layer exists in this model."""
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("q_layer.")
        )
