"""Matched-warm feedback ablation (Addendum A of
docs/research/QAE_PRIOR_BASELINES_PROTOCOL.md). No API calls.

    python scripts/qae/run_matched_warm.py [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/qae_prior_baselines/matched_warm.csv"
CONDITIONS = {"reference": (4, "TFIM", 8), "budget_b4": (4, "TFIM", 4),
              "budget_b16": (4, "TFIM", 16), "qubits_n6": (6, "TFIM", 8),
              "qubits_n8": (8, "TFIM", 8), "hamiltonian_xxz": (4, "XXZ", 8),
              "model_alt": (4, "TFIM", 8)}
FIELDS = ["condition", "seed", "arm", "source_seed", "phase", "order", "candidate",
          "fallback", "val_fid", "test_fid"]


def _closed(key: str, seed: int) -> dict:
    d = ROOT / ("outputs/qae_tfim_neutral_v5" if key == "reference"
                else f"outputs/qae_robustness/{key}")
    return json.loads((d / f"llm_closed_seed{seed}.json").read_text())


def _work(task):
    import torch
    torch.set_num_threads(1)
    from llm_vqc.experiments.qae_baselines.fast_eval import evaluate_fast
    from llm_vqc.experiments.qae_robustness.conditions import state_sets, train_seed
    meta, arch = task
    n, family = CONDITIONS[meta["condition"]][:2]
    train, val, test = state_sets(meta["seed"], family, n)
    r = evaluate_fast(arch, train, val, test, seed=train_seed(meta["seed"], arch), n=n)
    return {**meta, "val_fid": r.val_fid, "test_fid": r.test_fid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tasks = []
    for key, (n, family, budget) in CONDITIONS.items():
        w = budget // 2
        for seed in range(12):
            own = _closed(key, seed)
            other_seed = (seed + 6) % 12
            other = _closed(key, other_seed)
            for row, arch in zip(own["rows"], own["architectures"]):
                meta = {"condition": key, "seed": seed, "arm": "LLM-Closed",
                        "source_seed": seed, "phase": row["phase"], "order": row["order"],
                        "candidate": row["candidate"], "fallback": row["fallback"]}
                tasks.append((meta, arch))
            for row, arch in zip(other["rows"][:w], other["architectures"][:w]):
                meta = {"condition": key, "seed": seed, "arm": "LLM-Fresh",
                        "source_seed": other_seed, "phase": "fresh", "order": w + row["order"],
                        "candidate": row["candidate"], "fallback": row["fallback"]}
                tasks.append((meta, arch))
    print(f"{len(tasks)} evaluations", flush=True)
    with OUT.open("w", newline="") as fh, Pool(args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in pool.imap_unordered(_work, tasks, chunksize=2):
            writer.writerow(row)


if __name__ == "__main__":
    main()
