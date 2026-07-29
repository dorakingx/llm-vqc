"""Central registry of every bench_v2 search arm (contract C05).

Arms register here as they are implemented (Phases 3–4). The registry is
the single lookup the runners use — an arm not registered here cannot be
run, and the goal checker verifies every mandatory arm name appears.

Track A arms propose structure-only dicts in the space profile's grammar;
Track B arms propose complete candidates (structure + theta). Fixed
references are degenerate "arms" that emit their template once.
"""

from __future__ import annotations

TRACK_A_MANDATORY = (
    "random_structure",
    "evolutionary_structure",
    "greedy_growth",
    "llm_open_structure",
    "llm_archive_closed_structure",
)
TRACK_A_REFERENCES = (
    "ref_realamp_d1",
    "ref_realamp_d2",
    "ref_strongent_d1",
    "ref_strongent_d2",
)
TRACK_B_MANDATORY = (
    "random_joint",
    "evolutionary_joint",
    "llm_open_joint",
    "llm_closed_joint",
)

#: name -> factory; populated by llm_vqc.bench_v2.arms (Phase 3) and
#: llm_vqc.bench_v2.llm_arms (Phase 4) at import time.
ARM_FACTORIES: dict[str, object] = {}


class UnknownArmError(Exception):
    pass


def register_arm(name: str, factory) -> None:
    if name in ARM_FACTORIES:
        raise ValueError(f"arm {name!r} registered twice")
    ARM_FACTORIES[name] = factory


def get_arm_factory(name: str):
    try:
        return ARM_FACTORIES[name]
    except KeyError as exc:
        raise UnknownArmError(
            f"arm {name!r} is not registered; available: {sorted(ARM_FACTORIES)}"
        ) from exc
