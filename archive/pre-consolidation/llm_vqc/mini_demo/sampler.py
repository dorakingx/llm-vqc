"""Random-arm sampler over the mini demo's architecture space.

**Correction pass (section 10): sample uniformly over the 36 *canonical*
physical architectures, not the 54 raw field combinations.**

The raw grammar has 3 x 3 x 2 x 3 = 54 field combinations, but
`entangler_direction` has no physical effect for `CZ` or `NONE`
(`canonicalize_compact`), so those 54 combinations collapse to only 36
distinct physical circuits. Sampling each raw field uniformly and then
canonicalizing would over-weight CZ and NONE architectures (each drawn via
either of two directions) relative to the two physically-distinct CNOT
directions -- a biased distribution over the real architecture space. To
sample canonical architectures uniformly, we enumerate the 36 canonical
forms explicitly and draw one uniformly.

The four canonical entangler choices are:
  - CNOT 0->1  (control 0, target 1 -- physically distinct)
  - CNOT 1->0  (control 1, target 0 -- physically distinct)
  - CZ         (symmetric; direction canonicalized to 0_to_1)
  - NONE       (no gate; direction canonicalized to 0_to_1)
"""

from __future__ import annotations

import numpy as np

from llm_vqc.mini_demo.compact_schema import (
    ROTATION_GATE_CHOICES,
    CompactArchitecture,
    canonicalize_compact,
)

#: The four canonical (entangler, direction) pairs -- see module docstring.
CANONICAL_ENTANGLERS: tuple[tuple[str, str], ...] = (
    ("CNOT", "0_to_1"),
    ("CNOT", "1_to_0"),
    ("CZ", "0_to_1"),
    ("NONE", "0_to_1"),
)


def enumerate_canonical_architectures() -> list[CompactArchitecture]:
    """The exact list of 36 canonical physical architectures the demo can
    reach: 3 layer-1 gates x 4 canonical entanglers x 3 layer-2 gates.

    Every returned architecture is already canonical (idempotent under
    `canonicalize_compact`), so the list has no physical duplicates.
    """
    architectures: list[CompactArchitecture] = []
    for layer_1_gate in ROTATION_GATE_CHOICES:
        for entangler, direction in CANONICAL_ENTANGLERS:
            for layer_2_gate in ROTATION_GATE_CHOICES:
                architectures.append(
                    canonicalize_compact(
                        CompactArchitecture(
                            layer_1_gate=layer_1_gate,
                            entangler=entangler,
                            entangler_direction=direction,
                            layer_2_gate=layer_2_gate,
                        )
                    )
                )
    return architectures


#: Materialized once; sampling indexes into this uniform list.
CANONICAL_ARCHITECTURES: list[CompactArchitecture] = enumerate_canonical_architectures()


def sample_compact_architecture(rng: np.random.Generator) -> CompactArchitecture:
    """Draw one architecture uniformly from the 36 canonical physical
    architectures (not the 54 raw field combinations)."""
    index = int(rng.integers(0, len(CANONICAL_ARCHITECTURES)))
    return CANONICAL_ARCHITECTURES[index]
