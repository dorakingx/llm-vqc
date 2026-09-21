"""UCI HIGGS (dataset 280) loader with checksum + row-order verification and an
untouched external holdout.

Official semantics (UCI): 11,000,000 rows, column 0 = label (1 signal, 0
background), columns 1-21 = **low-level** kinematic/detector features, columns
22-28 = **high-level** physicist-engineered features; the **last 500,000 rows are
the official test set** (our external holdout). The loader verifies the row count
is exactly 11,000,000 and refuses to proceed otherwise (rather than inventing a
split).

Design: a single streaming pass over the gzip file (never loads 2.6 GB into
memory) writes a small cached `.npz` of (a) the first `DEV_BENCH_ROWS` non-holdout
rows for development + benchmark blocks and (b) a fixed `HOLDOUT_AUDIT_ROWS`
subset from the last 500,000 rows for the final external audit. The holdout cache
is written but must not be read until all design choices and primary results are
frozen (enforced by convention + the separate `load_holdout_audit` entry point).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache" / "uci_higgs"
GZ = CACHE / "HIGGS.csv.gz"
DEV_BENCH_NPZ = CACHE / "dev_benchmark_pool.npz"
HOLDOUT_NPZ = CACHE / "holdout_audit.npz"
MANIFEST = CACHE / "dataset_manifest.json"

N_ROWS_EXPECTED = 11_000_000
N_HOLDOUT = 500_000
HOLDOUT_START = N_ROWS_EXPECTED - N_HOLDOUT   # 10,500,000
DEV_BENCH_ROWS = 300_000                       # first N non-holdout rows we cache
HOLDOUT_AUDIT_ROWS = 20_000                    # fixed external-audit subset (first 20k of holdout)

LOW_LEVEL_NAMES = (
    "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
)  # 21 low-level (columns 1..21)
HIGH_LEVEL_NAMES = ("m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb")  # cols 22..28
N_LOW = 21
N_HIGH = 7
DATASET = {"name": "UCI HIGGS", "id": 280, "doi": "10.24432/C5V312",
           "license": "CC BY 4.0", "source": "https://archive.ics.uci.edu/dataset/280/higgs"}


def _sha256(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        read = 0
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if limit_bytes is not None and read + len(chunk) > limit_bytes:
                chunk = chunk[: limit_bytes - read]
            h.update(chunk); read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()


def prepare_cache(retrieval_date: str) -> dict:
    """One streaming pass: verify row count, cache dev/benchmark + holdout subsets,
    compute checksums, write the dataset manifest. Idempotent (skips if present)."""
    if DEV_BENCH_NPZ.exists() and HOLDOUT_NPZ.exists() and MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    if not GZ.exists():
        raise FileNotFoundError(f"{GZ} not found — download HIGGS first")

    import subprocess

    dev_rows, hold_rows = [], []
    total = 0
    hold_end = HOLDOUT_START + HOLDOUT_AUDIT_ROWS
    # Fast C decompression via `gzip -dc`, streamed line-by-line from stdout.
    proc = subprocess.Popen(["gzip", "-dc", str(GZ)], stdout=subprocess.PIPE, bufsize=1 << 20, text=True)
    try:
        for i, line in enumerate(proc.stdout):
            if i < DEV_BENCH_ROWS:
                dev_rows.append(np.array(line.split(","), dtype=np.float64))
            elif HOLDOUT_START <= i < hold_end:
                hold_rows.append(np.array(line.split(","), dtype=np.float64))
            total = i + 1
    finally:
        proc.stdout.close(); proc.wait()
    if total != N_ROWS_EXPECTED:
        raise RuntimeError(f"HIGGS row count {total} != official {N_ROWS_EXPECTED}; refusing to "
                           "proceed (cannot verify official split semantics).")

    dev = np.vstack(dev_rows); hold = np.vstack(hold_rows)
    assert dev.shape[1] == 1 + N_LOW + N_HIGH == 29, dev.shape
    np.savez_compressed(DEV_BENCH_NPZ, data=dev, row_start=0)
    np.savez_compressed(HOLDOUT_NPZ, data=hold, row_start=HOLDOUT_START)

    manifest = {
        "dataset": DATASET, "retrieval_date": retrieval_date,
        "compressed_sha256": _sha256(GZ),
        "compressed_bytes": GZ.stat().st_size,
        "row_count_verified": total, "expected_row_count": N_ROWS_EXPECTED,
        "n_columns": int(dev.shape[1]), "n_low_level": N_LOW, "n_high_level": N_HIGH,
        "label_definition": "column 0: 1 = signal (Higgs), 0 = background",
        "low_level_feature_names": list(LOW_LEVEL_NAMES),
        "high_level_feature_names": list(HIGH_LEVEL_NAMES),
        "holdout": {"convention": "last 500,000 rows = official test set",
                    "holdout_start_row": HOLDOUT_START,
                    "audit_subset_rows": HOLDOUT_AUDIT_ROWS,
                    "audit_subset_range": [HOLDOUT_START, HOLDOUT_START + HOLDOUT_AUDIT_ROWS]},
        "dev_benchmark_pool": {"rows": DEV_BENCH_ROWS, "range": [0, DEV_BENCH_ROWS],
                               "sha256_features": _sha256_array(dev)},
        "holdout_audit_sha256_features": _sha256_array(hold),
        "acquisition_command": "curl -o data/cache/uci_higgs/HIGGS.csv.gz "
                               "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        "cache_files": ["HIGGS.csv.gz (gitignored)", "dev_benchmark_pool.npz (gitignored)",
                        "holdout_audit.npz (gitignored)"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_dev_benchmark_pool(low_level_only: bool = True) -> dict:
    """Load the cached non-holdout pool. Returns labels + low-level (or all)
    features + global row indices (immutable IDs)."""
    d = np.load(DEV_BENCH_NPZ)["data"]
    labels = d[:, 0].astype(np.int64)
    low = d[:, 1:1 + N_LOW]
    high = d[:, 1 + N_LOW:]
    ids = np.arange(len(d))  # global row indices 0..DEV_BENCH_ROWS-1
    feats = low if low_level_only else d[:, 1:]
    return {"features": feats, "labels": labels, "row_ids": ids,
            "low_level": low, "high_level": high}


def load_holdout_audit(low_level_only: bool = True) -> dict:
    """External-audit subset — call ONLY after the design is frozen."""
    z = np.load(HOLDOUT_NPZ); d = z["data"]; start = int(z["row_start"])
    labels = d[:, 0].astype(np.int64)
    low = d[:, 1:1 + N_LOW]
    feats = low if low_level_only else d[:, 1:]
    ids = np.arange(start, start + len(d))
    return {"features": feats, "labels": labels, "row_ids": ids}
