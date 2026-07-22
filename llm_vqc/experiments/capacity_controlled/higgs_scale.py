"""HIGGS data-scale qualification: new disjoint pool, blocks, nested training
subsets, and the three input representations.

The qualification pool is streamed from the verified official HIGGS file at rows
[QUAL_START, QUAL_START+QUAL_ROWS) — a region that is **disjoint by construction**
from every HIGGS-v1 region: the v1 development subset (pool rows [0, 60k)), the v1
benchmark blocks (drawn from [60k, 300k)), and the official external holdout
([10.5M, 11M)). Immutable IDs are global HIGGS row indices.

Per qualification block: a fixed validation set (2,000), a fixed internal-test set
(5,000), and a training pool from which the 500/2,000/5,000/10,000 training sets
are **deterministic nested prefixes** (500 ⊂ 2,000 ⊂ 5,000 ⊂ 10,000).
Preprocessing is fit on the training subset of the given size only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass

import numpy as np

from llm_vqc.experiments.capacity_controlled import higgs_data as HD

QUAL_START = 1_000_000          # disjoint from v1 dev [0,60k), blocks [60k,300k), holdout [10.5M,11M)
QUAL_ROWS = 200_000
QUAL_NPZ = HD.CACHE / "qualification_pool.npz"
QUAL_MANIFEST = HD.CACHE / "qualification_manifest.json"

N_BLOCKS = 5          # qualification blocks (indices 0..4)
DEV_BLOCK_INDEX = 5   # extra block used ONLY for the development training-protocol audit
N_BLOCKS_TOTAL = N_BLOCKS + 1
TRAIN_SIZES = (500, 2000, 5000, 10000)
MAX_TRAIN = max(TRAIN_SIZES)
N_VAL = 2000
N_TEST = 5000
BLOCK_SEED = 20260720
PCA8, PCA16 = 8, 16
REPRESENTATIONS = ("R1_raw21", "R2_pca8", "R3_pca16")


def prepare_qualification_cache() -> dict:
    """Stream the verified gz once (early-exit) and cache the qualification rows."""
    if QUAL_NPZ.exists() and QUAL_MANIFEST.exists():
        return json.loads(QUAL_MANIFEST.read_text())
    if not HD.GZ.exists():
        raise FileNotFoundError(f"{HD.GZ} missing")
    end = QUAL_START + QUAL_ROWS
    rows = []
    proc = subprocess.Popen(["gzip", "-dc", str(HD.GZ)], stdout=subprocess.PIPE, bufsize=1 << 20, text=True)
    try:
        for i, line in enumerate(proc.stdout):
            if i >= end:
                break
            if i >= QUAL_START:
                rows.append(np.array(line.split(","), dtype=np.float64))
    finally:
        proc.stdout.close(); proc.kill(); proc.wait()
    data = np.vstack(rows)
    if len(data) != QUAL_ROWS:
        raise RuntimeError(f"qualification cache got {len(data)} rows, expected {QUAL_ROWS}")
    np.savez_compressed(QUAL_NPZ, data=data, row_start=QUAL_START)
    manifest = {"source": "UCI HIGGS (id 280)", "compressed_sha256_of_source": None,
                "qualification_row_range": [QUAL_START, end], "rows": int(len(data)),
                "disjoint_from": {"v1_dev": [0, 60000], "v1_blocks_region": [60000, 300000],
                                  "official_holdout": [HD.HOLDOUT_START, HD.N_ROWS_EXPECTED]},
                "features_sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
                "label_definition": "column 0: 1 = signal, 0 = background",
                "n_low_level": HD.N_LOW}
    QUAL_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_qualification_pool() -> dict:
    z = np.load(QUAL_NPZ); d = z["data"]; start = int(z["row_start"])
    return {"features": d[:, 1:1 + HD.N_LOW], "labels": d[:, 0].astype(np.int64),
            "row_ids": np.arange(start, start + len(d))}


@dataclass(frozen=True)
class QualBlock:
    index: int
    train_ids: tuple[int, ...]   # MAX_TRAIN, nested prefixes give the smaller sizes
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


def build_qual_blocks() -> list[QualBlock]:
    """5 mutually disjoint blocks; training pools are stratified and ordered so
    that prefixes of size 500/2000/5000/10000 are themselves stratified and nested."""
    pool = load_qualification_pool()
    ids, labels = pool["row_ids"], pool["labels"]
    rng = np.random.default_rng(BLOCK_SEED)
    queues = {c: list(ids[labels == c][rng.permutation(int((labels == c).sum()))]) for c in (0, 1)}
    ratio_pos = float((labels == 1).mean())

    def draw(n):
        n_pos = int(round(n * ratio_pos))
        out = [queues[1].pop() for _ in range(n_pos)] + [queues[0].pop() for _ in range(n - n_pos)]
        return out

    blocks = []
    for b in range(N_BLOCKS_TOTAL):
        # build the training pool as nested stratified prefixes
        train: list[int] = []
        prev = 0
        for size in TRAIN_SIZES:
            train += draw(size - prev)   # extend; prefix of length `size` stays stratified
            prev = size
        val = draw(N_VAL); test = draw(N_TEST)
        blocks.append(QualBlock(index=b, train_ids=tuple(int(x) for x in train),
                                val_ids=tuple(int(x) for x in val), test_ids=tuple(int(x) for x in test)))
    return blocks


def fit_representation(train_X: np.ndarray, representation: str):
    """Fit the representation on TRAINING data only. Returns (transform_fn, out_dim)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_X)
    if representation == "R1_raw21":
        return (lambda X: scaler.transform(X)), train_X.shape[1]
    n = PCA8 if representation == "R2_pca8" else PCA16
    pca = PCA(n_components=n, random_state=0).fit(scaler.transform(train_X))
    return (lambda X: pca.transform(scaler.transform(X))), n


def block_condition(block: QualBlock, train_size: int, representation: str):
    """Materialize (Xtr, ytr, Xva, yva, Xte, yte) for one (block, size, representation)."""
    pool = load_qualification_pool()
    id_to_pos = {int(r): i for i, r in enumerate(pool["row_ids"])}
    X, y = pool["features"], pool["labels"].astype(np.float64)

    def take(idlist):
        pos = [id_to_pos[i] for i in idlist]
        return X[pos], y[pos]

    tr_ids = block.train_ids[:train_size]          # nested prefix
    Xtr, ytr = take(tr_ids)
    Xva, yva = take(block.val_ids)
    Xte, yte = take(block.test_ids)
    tf, dim = fit_representation(Xtr, representation)   # train-only fit
    return {"Xtr": tf(Xtr), "ytr": ytr, "Xva": tf(Xva), "yva": yva, "Xte": tf(Xte), "yte": yte,
            "dim": dim, "train_ids": tr_ids}
