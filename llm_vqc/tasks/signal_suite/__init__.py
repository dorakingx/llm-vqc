"""bench_v2 signal-suite task package (versioned: signal_suite_v1).

Four physically interpretable 1-D signal families generated at exactly
`2**n_qubits` grid points for amplitude encoding — protocol §3 of
`docs/research/BENCHMARK_V2_PROTOCOL.md`. No MNIST/digits anywhere.
"""

from llm_vqc.tasks.signal_suite.config import SIGNAL_SUITE_VERSION
from llm_vqc.tasks.signal_suite.task import (
    ALLOWED_PROFILES,
    FAMILIES,
    SignalProfile,
    SignalSuiteTask,
    get_legacy_gauss_peak_task,
    get_signal_task,
    task_name,
)

__all__ = [
    "ALLOWED_PROFILES",
    "FAMILIES",
    "SIGNAL_SUITE_VERSION",
    "SignalProfile",
    "SignalSuiteTask",
    "get_legacy_gauss_peak_task",
    "get_signal_task",
    "task_name",
]
