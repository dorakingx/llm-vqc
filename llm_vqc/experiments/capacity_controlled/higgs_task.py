"""HIGGS capacity-controlled task: 10 disjoint blocks, train-only StandardScaler
+ PCA(8), and the controlled classification task spec (53 total trainable params).

Blocks are the primary statistical cluster. Construction is deterministic and
disjoint: the cached dev/benchmark pool (first 300k non-holdout rows) is split
into a **qualification region** (first `QUAL_ROWS` rows, used only by the
classical qualification gate) and a **benchmark region** (the rest, used only for
the 10 blocks). Within the benchmark region, rows are dealt into 10 blocks ×
{500 train, 250 val, 2000 test}, class-stratified, by popping from per-class
deterministic queues — so no sample appears in two blocks or two partitions.
Immutable IDs are the global HIGGS row indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_vqc.experiments.capacity_controlled import higgs_data as HD
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.tasks.base import DataSplit, FittedPreprocessing, TaskSpec, TrainValData, assert_disjoint_splits

QUAL_ROWS = 60_000            # first N pool rows reserved for classical qualification
N_BLOCKS = 10
N_TRAIN, N_VAL, N_TEST = 500, 250, 2000
PCA_COMPONENTS = 8
BLOCK_SEED = 20260719
_EPS = 1e-12

RAW_FEATURE_DIM = PCA_COMPONENTS               # 8
EMBED_PARAMS = RAW_FEATURE_DIM * S.N_QUBITS + S.N_QUBITS   # 36
HEAD_PARAMS = S.N_QUBITS + 1                    # 5
QUANTUM_PARAMS = S.QUANTUM_PARAM_COUNT          # 12
TOTAL_TRAINABLE_PARAMS = EMBED_PARAMS + QUANTUM_PARAMS + HEAD_PARAMS   # 53

SELECTION_METRIC = "logloss"
PRIMARY_TEST_METRIC = "logloss"
EQUIVALENCE_MARGIN = 0.01
TASK_NAME = "HIGGS"

SPEC = TaskSpec(name="HIGGS_low_level", description="UCI HIGGS 21 low-level -> PCA(8) binary",
                raw_feature_dim=RAW_FEATURE_DIM, classical_head_out_dim=1,
                metric_name=SELECTION_METRIC, lower_is_better=True, loss_name="bce")


def _fit_preprocessing(train_feats: np.ndarray) -> FittedPreprocessing:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_feats)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(scaler.transform(train_feats))

    def _transform(raw: np.ndarray, params: dict) -> np.ndarray:
        return params["pca"].transform(params["scaler"].transform(raw))

    return FittedPreprocessing(kind="standardize_then_pca8",
                               params={"scaler": scaler, "pca": pca}, _transform_fn=_transform)


@dataclass(frozen=True)
class BlockAssignment:
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


def build_block_assignments() -> list[BlockAssignment]:
    """Deterministic, disjoint, class-stratified assignment of benchmark-region
    rows into 10 blocks. Uses immutable global row IDs."""
    pool = HD.load_dev_benchmark_pool(low_level_only=True)
    ids, labels = pool["row_ids"], pool["labels"]
    bench_mask = ids >= QUAL_ROWS
    bench_ids = ids[bench_mask]
    bench_labels = labels[bench_mask]

    rng = np.random.default_rng(BLOCK_SEED)
    # per-class deterministic queues
    queues = {}
    for c in (0, 1):
        cls_ids = bench_ids[bench_labels == c]
        queues[c] = list(cls_ids[rng.permutation(len(cls_ids))])
    # per-block, per-partition stratified counts (preserve overall class ratio)
    ratio_pos = float((bench_labels == 1).mean())

    def _draw(n):
        n_pos = int(round(n * ratio_pos)); n_neg = n - n_pos
        out = [queues[1].pop() for _ in range(n_pos)] + [queues[0].pop() for _ in range(n_neg)]
        return tuple(int(x) for x in out)

    blocks = []
    for _ in range(N_BLOCKS):
        blocks.append(BlockAssignment(train_ids=_draw(N_TRAIN), val_ids=_draw(N_VAL), test_ids=_draw(N_TEST)))
    return blocks


def _pool_by_id():
    pool = HD.load_dev_benchmark_pool(low_level_only=True)
    id_to_pos = {int(rid): i for i, rid in enumerate(pool["row_ids"])}
    return pool, id_to_pos


def build_block(block_index: int, assignments: list[BlockAssignment] | None = None):
    """Return (TrainValData, test DataSplit) for one block, preprocessing fit on
    that block's TRAIN partition only (StandardScaler -> PCA8)."""
    assignments = assignments or build_block_assignments()
    a = assignments[block_index]
    pool, id_to_pos = _pool_by_id()
    X, y = pool["features"], pool["labels"]

    def _raw(ids):
        pos = [id_to_pos[i] for i in ids]
        return X[pos], y[pos].astype(np.float64)

    Xtr, ytr = _raw(a.train_ids); Xva, yva = _raw(a.val_ids); Xte, yte = _raw(a.test_ids)
    prep = _fit_preprocessing(Xtr)
    train = DataSplit(features=prep.transform(Xtr), targets=ytr,
                      sample_ids=tuple(f"HIGGS-b{block_index}-tr-{i}" for i in a.train_ids))
    val = DataSplit(features=prep.transform(Xva), targets=yva,
                    sample_ids=tuple(f"HIGGS-b{block_index}-va-{i}" for i in a.val_ids))
    test = DataSplit(features=prep.transform(Xte), targets=yte,
                     sample_ids=tuple(f"HIGGS-b{block_index}-te-{i}" for i in a.test_ids))
    assert_disjoint_splits(train, val, test)
    tv = TrainValData(spec=SPEC, train=train, val=val, split_seed=BLOCK_SEED + block_index, preprocessing=prep)
    return tv, test


def qualification_split(val_fraction: float = 0.25, seed: int = 0):
    """Development-only qualification data (train/val) from the qualification
    region — disjoint from every block and from the holdout. Returns raw 21
    low-level features (qualification models do their own preprocessing)."""
    pool = HD.load_dev_benchmark_pool(low_level_only=True)
    ids, X, y = pool["row_ids"], pool["features"], pool["labels"]
    mask = ids < QUAL_ROWS
    Xq, yq = X[mask], y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xq))
    n_val = int(len(Xq) * val_fraction)
    va, tr = perm[:n_val], perm[n_val:]
    return {"train_X": Xq[tr], "train_y": yq[tr], "val_X": Xq[va], "val_y": yq[va],
            "train_ids": ids[mask][tr], "val_ids": ids[mask][va]}


def total_trainable_params(ir) -> int:
    from llm_vqc.evaluation.model import HybridQNNModel
    m = HybridQNNModel(ir, raw_feature_dim=RAW_FEATURE_DIM, head_out_dim=1)
    return int(sum(p.numel() for p in m.parameters()))


def assert_capacity(ir) -> int:
    total = total_trainable_params(ir)
    if total != TOTAL_TRAINABLE_PARAMS:
        from llm_vqc.ir.canonicalize import structural_hash
        raise RuntimeError(f"HIGGS CAPACITY VIOLATION: {total} != {TOTAL_TRAINABLE_PARAMS} "
                           f"(hash {structural_hash(ir)[:12]})")
    return total


def evaluate_all_test_metrics(ir, classical_state: dict, test, with_curves: bool = False) -> dict:
    """Reload an already-trained model (no retrain) and compute the full protected-
    test classification metric set on HIGGS PCA(8) features."""
    import torch

    from llm_vqc.evaluation.model import HybridQNNModel
    from llm_vqc.experiments.capacity_controlled.t2_task import full_classification_metrics
    model = HybridQNNModel(ir, raw_feature_dim=RAW_FEATURE_DIM, head_out_dim=1)
    reloaded = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
                for k, v in classical_state.items()}
    model.load_state_dict(reloaded)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(test.features, dtype=torch.float64)).cpu().numpy().reshape(-1)
    return full_classification_metrics(preds, test.targets, with_curves=with_curves)
