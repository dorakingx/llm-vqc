"""bench_v2 arm implementations. Importing this package registers every
classical arm in `llm_vqc.bench_v2.arm_registry` (LLM arms register from
`llm_vqc.bench_v2.llm_arms`, Phase 4)."""

from llm_vqc.bench_v2.arm_registry import register_arm
from llm_vqc.bench_v2.arms.joint_arms import EvolutionaryJointArm, RandomJointArm
from llm_vqc.bench_v2.arms.structure_arms import (
    EvolutionaryStructureArm,
    FixedReferenceArm,
    GreedyGrowthArm,
    RandomStructureArm,
)
from llm_vqc.bench_v2.space import REFERENCE_ARMS

register_arm("random_structure", RandomStructureArm)
register_arm("evolutionary_structure", EvolutionaryStructureArm)
register_arm("greedy_growth", GreedyGrowthArm)
for _ref in REFERENCE_ARMS:
    register_arm(_ref, FixedReferenceArm)
register_arm("random_joint", RandomJointArm)
register_arm("evolutionary_joint", EvolutionaryJointArm)

__all__ = [
    "EvolutionaryJointArm",
    "EvolutionaryStructureArm",
    "FixedReferenceArm",
    "GreedyGrowthArm",
    "RandomJointArm",
    "RandomStructureArm",
]
