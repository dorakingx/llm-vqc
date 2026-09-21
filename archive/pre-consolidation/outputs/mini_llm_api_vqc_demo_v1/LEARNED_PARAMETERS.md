# Learned Parameters -- mini_llm_api_vqc_demo_v1

Initialization policy: `mini_demo_init_v1`. Differentiation method: `backprop`. dtype: float64. device: CPU.

## Gradient-trained parameters (AdamW, 5 epochs, batch size 16)

### Classical embedding: `Linear(21 -> 2)` -- 42 weights + 2 biases
- initialization: weights Xavier (Glorot) uniform, biases zeros.

### Quantum parameters: 4 rotation angles (2 per variational layer)
- initialization: `Uniform[-pi, pi]`.
- optimized jointly with the classical parameters through PennyLane's
  `qml.qnn.TorchLayer` with `diff_method="backprop"`.

### Classical output head: `Linear(2 -> 1)` -- 2 weights + 1 bias
- initialization: weights Xavier uniform, bias zero.

**Total: 44 + 4 + 3 = 51 trainable parameters** -- every candidate is
hard-verified before training (`verify_fixed_capacity`) and again inside
`mini_demo_init_v1` (exactly 4 quantum / 51 total, all float64, all CPU).

## Not gradient-trained (chosen by the LLM / random arm, or fixed by design)
- Gate families for layer 1 and layer 2 (`RX`/`RY`/`RZ`).
- The optional entangler (`CNOT`/`CZ`/`NONE`) and its direction.
- Layer ordering; the input-dependent RY encoding angles; the fixed
  encoding (RY on wires [0,1]) and measurement (Z on wires [0,1]);
  dataset values and targets.

## Initial vs learned quantum angles (each selected arm)

### Random
- initial quantum angles: `[1.625463, 1.510458, -3.085431, 2.306071]`
- learned quantum angles: `[2.088219, 1.056408, -2.044101, 1.852001]`
- at least one angle changed (> 1e-6): True

### LLM Open-loop
- initial quantum angles: `[2.077876, -2.449346, 3.065049, -0.724186]`
- learned quantum angles: `[2.292381, -3.027945, 3.512008, -0.118235]`
- at least one angle changed (> 1e-6): True

### LLM Closed-loop
- initial quantum angles: `[1.427053, 2.849148, 0.744705, -2.910924]`
- learned quantum angles: `[2.260157, 3.019292, 0.585384, -3.366113]`
- at least one angle changed (> 1e-6): True
