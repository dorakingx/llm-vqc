"""Dataset manifests and required data checks (protocol §3).

`dataset_manifest` produces the JSON blob every bench_v2 cell stores next
to its results: generation hashes (deterministic-regeneration evidence),
disjointness confirmation, target statistics, degeneracy guards, and the
frozen generator constants. `verify_deterministic_regeneration` is the
runtime check that two independent builds from the same seed are
byte-identical — the same property the test suite locks in CI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from llm_vqc.tasks.base import DataSplit, assert_disjoint_splits
from llm_vqc.tasks.signal_suite.config import SIGNAL_SUITE_VERSION
from llm_vqc.tasks.signal_suite.task import SignalSuiteTask


def split_hash(split: DataSplit) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(split.features, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(split.targets, dtype=np.float64).tobytes())
    digest.update("\x00".join(split.sample_ids).encode())
    return digest.hexdigest()


def dataset_manifest(task: SignalSuiteTask, data_seed: int) -> dict:
    train_val, build_diag = task.build(data_seed)
    test, test_diag = task.build_test(data_seed)
    assert_disjoint_splits(train_val.train, train_val.val, test)

    generator_blob = json.dumps(asdict(task.generator_config), sort_keys=True)
    manifest = {
        "signal_suite_version": SIGNAL_SUITE_VERSION,
        "task_name": task.spec.name,
        "family": task.profile.family,
        "n_qubits": task.profile.n_qubits,
        "feature_count": task.profile.feature_count,
        "data_seed": data_seed,
        "split_seed": train_val.split_seed,
        "sizes": {
            "train": len(train_val.train), "val": len(train_val.val), "test": len(test),
        },
        "split_hashes": {
            "train": split_hash(train_val.train),
            "val": split_hash(train_val.val),
            "test": split_hash(test),
        },
        "generator_config": json.loads(generator_blob),
        "generator_config_sha256": hashlib.sha256(generator_blob.encode()).hexdigest(),
        "diagnostics": {**build_diag, **test_diag},
        "splits_disjoint": True,
    }
    return manifest


def verify_deterministic_regeneration(task: SignalSuiteTask, data_seed: int) -> bool:
    """Two independent full regenerations must be byte-identical."""
    first = dataset_manifest(task, data_seed)
    second = dataset_manifest(task, data_seed)
    return first["split_hashes"] == second["split_hashes"]
