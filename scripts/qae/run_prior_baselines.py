"""Run the prior-matched baseline study (no API calls).

    python scripts/qae/run_prior_baselines.py [--workers 4]

Writes outputs/qae_prior_baselines/candidate_results.csv (resumable).
"""
from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/qae_prior_baselines"
CONDITIONS = [("reference", 4, "TFIM"), ("hamiltonian_xxz", 4, "XXZ"),
              ("qubits_n6", 6, "TFIM"), ("qubits_n8", 8, "TFIM")]
FIELDS = ["condition", "seed", "arm", "candidate", "order", "val_fid", "test_fid",
          "train_fid", "architecture"]


def _work(task):
    import json

    import torch
    torch.set_num_threads(1)
    from llm_vqc.experiments.qae_baselines.fast_eval import evaluate_fast
    from llm_vqc.experiments.qae_robustness.conditions import state_sets, train_seed

    key, n, family, seed, arm, order, name, arch = task
    train, val, test = state_sets(seed, family, n)
    r = evaluate_fast(arch, train, val, test, seed=train_seed(seed, arch), n=n)
    return {"condition": key, "seed": seed, "arm": arm, "candidate": name, "order": order,
            "val_fid": r.val_fid, "test_fid": r.test_fid, "train_fid": r.train_fid,
            "architecture": json.dumps(arch, separators=(",", ":"))}


def main() -> None:
    from llm_vqc.experiments.qae_baselines.arms import ARMS, pool_for

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "candidate_results.csv"
    done = set()
    if path.exists():
        with path.open() as fh:
            done = {(r["condition"], int(r["seed"]), r["arm"], int(r["order"]))
                    for r in csv.DictReader(fh)}
    tasks = []
    for key, n, family in CONDITIONS:  # cheapest conditions first
        for seed in range(12):
            for arm in ARMS:
                for order, (name, arch) in enumerate(pool_for(arm, key, n, family, seed), 1):
                    if (key, seed, arm, order) not in done:
                        tasks.append((key, n, family, seed, arm, order, name, arch))
    print(f"{len(tasks)} evaluations to run ({len(done)} already done)", flush=True)
    new = not path.exists()
    with path.open("a", newline="") as fh, Pool(args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
            writer.writerow(row)
            if i % 200 == 0:
                fh.flush()
                print(f"{i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
