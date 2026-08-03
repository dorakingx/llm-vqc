"""Angle ranges: how a proposer expresses where it thinks good angles are.

An LLM cannot sample a continuum. Measured three times on gpt-5-nano: it
collapses onto pi/4 and friends, then onto whatever examples the prompt
shows, then onto typeable digit runs like 0.123. At best 59% of its
angles were distinct where a uniform sampler gives 100%. Asking it for a
number is asking it to do the one thing it structurally cannot do.

So it does not choose a number. It chooses a RANGE from a finite menu and
the harness draws uniformly inside that range. Selecting from a fixed
list is what a constrained-decoding model is reliably good at, and the
angle that finally runs is still continuous.

The mechanism contains the random baseline exactly: choosing bins
uniformly and then sampling inside the chosen bin is the same
distribution as sampling uniformly over the whole interval. So the
comparison is precisely "does biasing the bin choice help?", and a
proposer that has no opinion can always pick ANY and lose nothing.
"""

from __future__ import annotations

import numpy as np

#: Eight equal bins over [-pi, pi], plus an explicit opt-out. The labels
#: are the numbers themselves so the menu is self-documenting in the
#: prompt and in any logged proposal.
N_BINS = 8
ANY_RANGE = "any"

_EDGES = np.linspace(-np.pi, np.pi, N_BINS + 1)
ANGLE_RANGES: tuple[str, ...] = tuple(
    f"[{_EDGES[i]:.3f},{_EDGES[i + 1]:.3f}]" for i in range(N_BINS)
) + (ANY_RANGE,)

_BOUNDS: dict[str, tuple[float, float]] = {
    label: (float(_EDGES[i]), float(_EDGES[i + 1]))
    for i, label in enumerate(ANGLE_RANGES[:-1])
}
_BOUNDS[ANY_RANGE] = (float(-np.pi), float(np.pi))


def bounds(label: str) -> tuple[float, float]:
    if label not in _BOUNDS:
        raise KeyError(f"unknown angle range {label!r}")
    return _BOUNDS[label]


def draw(rng: np.random.Generator, label: str) -> float:
    """The harness draws the angle, never the proposer, so a proposer
    cannot smuggle a hand-picked value through the range mechanism."""
    lo, hi = bounds(label)
    return float(rng.uniform(lo, hi))


def uniform_label(rng: np.random.Generator) -> str:
    """What a proposer with no opinion picks. Drawing a bin uniformly and
    then drawing inside it reproduces a uniform draw over [-pi, pi]."""
    return str(rng.choice(ANGLE_RANGES[:-1]))


def resolve(ops: list[dict], rng: np.random.Generator) -> list[dict]:
    """Turn `theta_range` into a concrete `theta`, in place of a copy.

    Operations that already carry a numeric `theta` are passed through, so
    the same code serves both the range-based arms and the reference
    ansatz, which fixes its own angles.
    """
    out = []
    for op in ops:
        item = dict(op)
        label = item.pop("theta_range", None)
        if item.get("gate") == "H":
            # A strict JSON schema cannot say "null only when gate is H",
            # so proposers do attach a range to H. Normalise here rather
            # than failing them for a rule the schema could not express.
            item["theta"] = None
        elif item.get("theta") is None and label is not None:
            item["theta"] = draw(rng, label)
        out.append(item)
    return out
