#!/usr/bin/env python
"""Run the HIGGS-v1 primary matrix: 10 disjoint blocks x 3 seeds x 3 arms (B=25)
+ classical baselines C0-C7 + paired quantum ablations (freeze/train, product/
entangled). Selection + primary test metric = log-loss. Fails loudly if any
primary candidate != 53 trainable params. Resumable (per-block stores + aux JSON).

No LLM API. External holdout is NOT touched here (separate audit script).

Usage:
  python scripts/capacity_controlled/run_experiment_higgs.py --blocks 0-9 --seeds 0,1,2 --abl-seeds 0,1 --budget 25 --tag pilot
  python scripts/capacity_controlled/run_experiment_higgs.py --blocks 0 --seeds 0 --abl-seeds 0 --budget 5 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import higgs_ablations as HA
from llm_vqc.experiments.capacity_controlled import higgs_baselines as HB
from llm_vqc.experiments.capacity_controlled import higgs_task as HT
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.arms import (
    ControlledEvolutionaryArm, ControlledEvoConfig, ControlledGreedyArm, ControlledRandomArm)
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init, init_policy_metadata
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.search.runner import SearchRunner

ROOT = Path(__file__).resolve().parents[2]
GREEDY_K = 5
EVO_CFG = ControlledEvoConfig(mu=5, lambda_=5)
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")


def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _make_arm(name):
    if name == "controlled_random":
        return ControlledRandomArm(lower_is_better=True)
    if name == "controlled_evolutionary":
        return ControlledEvolutionaryArm(lower_is_better=True, config=EVO_CFG)
    if name == "controlled_greedy":
        return ControlledGreedyArm(lower_is_better=True, k=GREEDY_K)
    raise ValueError(name)


def run_search_cell(db, arm, block, seed, budget, tv, test, cfg, git_sha):
    run_id = f"{arm}_b{block}_s{seed}"
    store = ResultStore.open_or_create(
        db, run_id=run_id, config_json="{}",
        config_reproducibility_fields={"space": "HIGGS_capacity_controlled_v1", "arm": arm,
                                       "budget": budget, "block": block, "repetition_seed": seed,
                                       "training": cfg.model_dump(), "init_policy": init_policy_metadata()["name"]},
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat())
    runner = SearchRunner(_make_arm(arm), HT.TASK_NAME, tv, cfg, budget_limit=budget, run_seed=seed,
                          result_store=store, run_id=run_id, git_sha=git_sha,
                          extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    result = runner.run()
    row = {"run_id": run_id, "arm": arm, "block": block, "seed": seed,
           "ledger_summary": result.ledger_summary,
           "selected_structural_hash": result.selected_structural_hash,
           "selected_val_logloss": result.selected_val_metric_value, "test": None}
    h, ts = result.selected_structural_hash, result.selected_train_seed
    if h is not None:
        w = store.get_trained_weights(HT.TASK_NAME, h, ts); c = store.get_cached(HT.TASK_NAME, h, ts)
        if w and c and c.circuit_canonical_json:
            ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
            HT.assert_capacity(ir)
            row["test"] = HT.evaluate_all_test_metrics(ir, w["classical_state"], test)
    store.close()
    return row


def _train_test(ir, tv, test, train_seed, freeze_quantum=False):
    HT.assert_capacity(ir)
    out = train_model(ir, tv, TrainingConfig(), train_seed, init_policy=apply_explicit_init,
                      freeze_quantum=freeze_quantum)
    if not out.success:
        return {"logloss": None, "error": out.error_message}
    md = HT.evaluate_all_test_metrics(ir, out.trained_classical_state, test)
    md["val_logloss"] = out.final_val_metric
    return md


def run_aux(block, seeds, abl_seeds, tv, test, aux_path: Path):
    aux = json.loads(aux_path.read_text()) if aux_path.exists() else {}
    aux.setdefault("classical", {})
    for seed in seeds:
        if str(seed) not in aux["classical"]:
            aux["classical"][str(seed)] = HB.run_all_classical(tv, test, seed=seed)
            aux_path.write_text(json.dumps(aux, indent=2))

    if "freeze_train" not in aux:
        rows = []
        for ai, g in enumerate(HA.predeclared_architectures()):
            ir = S.genome_to_ir(g); h = structural_hash(ir)
            for seed in abl_seeds:
                ts = train_seed_for_circuit(seed, h)  # identical init for both variants
                tr = _train_test(ir, tv, test, ts, freeze_quantum=False)
                fr = _train_test(ir, tv, test, ts, freeze_quantum=True)
                rows.append({"arch": ai, "block": block, "seed": seed,
                             "trainable_logloss": tr["logloss"], "frozen_logloss": fr["logloss"]})
        aux["freeze_train"] = rows
        aux_path.write_text(json.dumps(aux, indent=2))

    if "product_entangled" not in aux:
        rows = []
        for pi, (e, p) in enumerate(HA.entangled_pairs(20)):
            ire, irp = S.genome_to_ir(e), S.genome_to_ir(p)
            diag = HA.pair_diagnostics(e, p)
            for seed in abl_seeds:
                pair_seed = derive_child_seed(seed, "higgs_pairB", str(pi))  # identical init both
                re = _train_test(ire, tv, test, pair_seed)
                rp = _train_test(irp, tv, test, pair_seed)
                rows.append({"pair": pi, "block": block, "seed": seed,
                             "entangled_logloss": re["logloss"], "product_logloss": rp["logloss"],
                             "depth_diff": diag["depth_diff"], "param_count_preserved": diag["param_count_preserved"]})
        aux["product_entangled"] = rows
        aux_path.write_text(json.dumps(aux, indent=2))
    return aux


def _parse_blocks(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0-9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--abl-seeds", default="0,1")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_higgs_v1"))
    args = ap.parse_args()
    blocks = _parse_blocks(args.blocks); seeds = [int(x) for x in args.seeds.split(",")]
    abl_seeds = [int(x) for x in args.abl_seeds.split(",")]
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainingConfig(); git_sha = _git_sha()

    print("=" * 72)
    print(f"HIGGS-v1 tag={args.tag} blocks={blocks} seeds={seeds} abl_seeds={abl_seeds} budget={args.budget}")
    print(f"controlled: 4q angle-RY Z-all, {HT.TOTAL_TRAINABLE_PARAMS} params; select+primary=log-loss")
    print("=" * 72)

    assignments = HT.build_block_assignments()
    all_runs, all_aux = [], {}
    for block in blocks:
        tv, test = HT.build_block(block, assignments)
        bd = run_dir / f"block_{block}"; bd.mkdir(parents=True, exist_ok=True)
        db = bd / "results.sqlite"
        for arm in ARMS:
            for seed in seeds:
                row = run_search_cell(db, arm, block, seed, args.budget, tv, test, cfg, git_sha)
                all_runs.append(row)
                ll = row["test"]["logloss"] if row["test"] else None
                print(f"[{row['run_id']}] val_ll={row['selected_val_logloss']} test_ll={ll}")
        all_aux[str(block)] = run_aux(block, seeds, abl_seeds, tv, test, bd / "aux_results.json")
        print(f"[block {block}] aux done")

    summary = {"tag": args.tag, "blocks": blocks, "seeds": seeds, "abl_seeds": abl_seeds,
               "budget": args.budget, "git_sha": git_sha, "arms": list(ARMS),
               "block_assignments": [{"block": i, "train_ids": list(a.train_ids), "val_ids": list(a.val_ids),
                                      "test_ids": list(a.test_ids)} for i, a in enumerate(assignments)],
               "runs": all_runs, "aux": all_aux}
    (run_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {run_dir}/{args.tag}_summary.json")


if __name__ == "__main__":
    main()
