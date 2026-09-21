#!/usr/bin/env python
"""HIGGS data-scale qualification study (two phases, resumable).

Phase A — development-only training-protocol audit: grid over lr x epochs x
weight-decay x early-stopping on the DEV block (never a qualification block's test
labels), selected by validation log-loss.

Phase B — factorial qualification: representation {R1 raw21, R2 PCA8, R3 PCA16} x
training size {500, 2000, 5000, 10000} x model {M0..M6} x 5 qualification blocks
x 5 predeclared VQC architecture seeds, using the selected protocol.

No architecture search. No LLM API. Official holdout untouched.

Usage:
  python scripts/capacity_controlled/run_qualification.py --phase A
  python scripts/capacity_controlled/run_qualification.py --phase B
  python scripts/capacity_controlled/run_qualification.py --phase B --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from llm_vqc.experiments.capacity_controlled import higgs_qual_models as QM
from llm_vqc.experiments.capacity_controlled import higgs_scale as HS
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.higgs_hardware import transpile_metrics
from llm_vqc.evaluation.seeds import derive_child_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "higgs_data_scale_qualification_v1"
RUN = ROOT / "runs" / "higgs_data_scale_qualification_v1"

# Phase A (predeclared): dev block, R2 representation, size 2000, 2 entangled archs
AUDIT_REP = "R2_pca8"
AUDIT_SIZE = 2000
AUDIT_ARCHS = (5000, 5001)
GRID_LR = (0.005, 0.01, 0.05)
GRID_EPOCHS = (20, 50)
GRID_WD = (1e-5, 1e-3)
GRID_ES = (False, True)


def phase_a():
    RUN.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "training_protocol_search.csv"
    blocks = HS.build_qual_blocks()
    dev = blocks[HS.DEV_BLOCK_INDEX]
    cond = HS.block_condition(dev, AUDIT_SIZE, AUDIT_REP)
    rows = []
    for arch_seed in AUDIT_ARCHS:
        ir = S.genome_to_ir(QM.entangled_arch(arch_seed))
        for lr in GRID_LR:
            for ep in GRID_EPOCHS:
                for wd in GRID_WD:
                    for es in GRID_ES:
                        proto = QM.Protocol(lr=lr, epochs=ep, weight_decay=wd, early_stopping=es)
                        seed = derive_child_seed(0, "audit", f"{arch_seed}_{proto.label()}")
                        r = QM.train_vqc(ir, cond["dim"], cond["Xtr"], cond["ytr"],
                                         cond["Xva"], cond["yva"], cond["Xte"], proto, seed)
                        vm = QM.metrics(r["val_pred"], cond["yva"])
                        rows.append({"arch_seed": arch_seed, "lr": lr, "epochs": ep, "weight_decay": wd,
                                     "early_stopping": es, "protocol": proto.label(),
                                     "val_logloss": vm["logloss"], "val_auroc": vm["auroc"],
                                     "epochs_run": r["epochs_run"], "failed": r["failed"],
                                     "runtime_s": round(r["runtime_s"], 2)})
                        print(f"  {proto.label():34s} arch{arch_seed} val_ll={vm['logloss']:.4f} "
                              f"val_auc={vm['auroc']:.4f} ep_run={r['epochs_run']} {r['runtime_s']:.1f}s")
    import csv
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # select by mean validation log-loss across the audit architectures
    agg = {}
    for r in rows:
        agg.setdefault(r["protocol"], []).append(r["val_logloss"])
    ranked = sorted(agg.items(), key=lambda kv: float(np.mean(kv[1])))
    best_label, best_vals = ranked[0]
    best_row = next(r for r in rows if r["protocol"] == best_label)
    sel = {"selected_protocol": {"lr": best_row["lr"], "epochs": best_row["epochs"],
                                 "weight_decay": best_row["weight_decay"],
                                 "early_stopping": best_row["early_stopping"], "batch_size": QM.BATCH},
           "label": best_label, "mean_val_logloss": float(np.mean(best_vals)),
           "selection_rule": "lowest mean validation log-loss across the predeclared audit architectures",
           "audit_setup": {"block": "development block (index 5, disjoint from all qualification blocks)",
                           "representation": AUDIT_REP, "train_size": AUDIT_SIZE, "arch_seeds": list(AUDIT_ARCHS)},
           "test_labels_used": False,
           "ranking_top5": [{"protocol": k, "mean_val_logloss": float(np.mean(v))} for k, v in ranked[:5]]}
    (OUT / "training_protocol_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nSELECTED: {best_label} (mean val log-loss {sel['mean_val_logloss']:.4f})")
    return sel


def _load_protocol() -> QM.Protocol:
    sel = json.loads((OUT / "training_protocol_selection.json").read_text())["selected_protocol"]
    return QM.Protocol(lr=sel["lr"], epochs=sel["epochs"], weight_decay=sel["weight_decay"],
                       early_stopping=sel["early_stopping"])


def phase_b(smoke=False, only_reps=None, only_blocks=None):
    """Shardable by (representation, block): each (rep, block) unit writes its own
    results file and is skipped if present, so independent workers can run in
    parallel without interfering. Results are deterministic per unit."""
    RUN.mkdir(parents=True, exist_ok=True)
    proto = _load_protocol()
    print(f"Using selected protocol: {proto.label()}")
    blocks = HS.build_qual_blocks()
    reps = HS.REPRESENTATIONS if not smoke else ("R2_pca8",)
    sizes = HS.TRAIN_SIZES if not smoke else (500,)
    qblocks = range(HS.N_BLOCKS) if not smoke else range(1)
    arch_seeds = QM.VQC_ARCH_SEEDS if not smoke else QM.VQC_ARCH_SEEDS[:2]
    if only_reps:
        reps = tuple(r for r in reps if r in only_reps)
    if only_blocks is not None:
        qblocks = [b for b in qblocks if b in only_blocks]

    for rep in reps:
        for b in qblocks:
            part = RUN / f"results_{rep}_b{b}.json"
            if part.exists() and not smoke:
                print(f"[skip] {part.name} exists"); continue
            rows = []
            t_block = time.perf_counter()
            for size in sizes:
                cond = HS.block_condition(blocks[b], size, rep)
                dim = cond["dim"]
                base = {"representation": rep, "block": b, "train_size": size, "input_dim": dim}
                # M0 class prior
                pv, pt, _ = QM.m0_class_prior(cond["Xtr"], cond["ytr"], cond["Xva"], cond["Xte"])
                rows.append({**base, "model": "M0_class_prior", "arch_seed": None, "total_params": 0,
                             "total_trainable_params": 0, "failed": False, "runtime_s": 0.0,
                             **{f"val_{k}": v for k, v in QM.metrics(pv, cond["yva"]).items()},
                             **{f"test_{k}": v for k, v in QM.metrics(pt, cond["yte"]).items()}})
                # M1 logreg
                t0 = time.perf_counter()
                pv, pt, npar = QM.m1_logreg(cond["Xtr"], cond["ytr"], cond["Xva"], cond["yva"], cond["Xte"])
                rows.append({**base, "model": "M1_logreg", "arch_seed": None, "total_params": npar,
                             "total_trainable_params": npar, "failed": False,
                             "runtime_s": round(time.perf_counter() - t0, 2),
                             **{f"val_{k}": v for k, v in QM.metrics(pv, cond["yva"]).items()},
                             **{f"test_{k}": v for k, v in QM.metrics(pt, cond["yte"]).items()}})
                # M2 MLP
                t0 = time.perf_counter()
                seed = derive_child_seed(b, "M2", f"{rep}_{size}")
                pv, pt, npar, r = QM.m2_mlp(cond["Xtr"], cond["ytr"], cond["Xva"], cond["yva"], cond["Xte"], proto, seed)
                rows.append({**base, "model": "M2_mlp", "arch_seed": None, "total_params": npar,
                             "total_trainable_params": npar, "failed": r["failed"],
                             "runtime_s": round(time.perf_counter() - t0, 2),
                             **{f"val_{k}": v for k, v in QM.metrics(pv, cond["yva"]).items()},
                             **{f"test_{k}": v for k, v in QM.metrics(pt, cond["yte"]).items()}})
                # M3/M4(=M6)/M5 VQC
                for a in arch_seeds:
                    prod_ir = S.genome_to_ir(QM.product_arch(a))
                    ent_ir = S.genome_to_ir(QM.entangled_arch(a))
                    for model_name, ir, frozen in (("M3_product", prod_ir, False),
                                                   ("M4_entangled", ent_ir, False),
                                                   ("M5_frozen", ent_ir, True)):
                        vseed = derive_child_seed(b, "vqc", f"{rep}_{size}_{a}")  # paired init M4/M5
                        r = QM.train_vqc(ir, dim, cond["Xtr"], cond["ytr"], cond["Xva"], cond["yva"],
                                         cond["Xte"], proto, vseed, freeze_quantum=frozen)
                        hw = transpile_metrics(ir)
                        rows.append({**base, "model": model_name, "arch_seed": a,
                                     **r["capacity"], "failed": r["failed"],
                                     "epochs_run": r["epochs_run"], "runtime_s": round(r["runtime_s"], 2),
                                     "q_unchanged": r["q_unchanged"],
                                     "transpiled_two_qubit": hw["transpiled_two_qubit_cx"],
                                     "transpiled_depth": hw["transpiled_depth"],
                                     "swap_est": hw["inserted_swaps_est"],
                                     "logical_two_qubit": hw["logical_two_qubit"],
                                     **{f"val_{k}": v for k, v in QM.metrics(r["val_pred"], cond["yva"]).items()},
                                     **{f"test_{k}": v for k, v in QM.metrics(r["test_pred"], cond["yte"]).items()}})
                print(f"  [{rep} b{b} n={size}] done ({time.perf_counter()-t_block:.0f}s elapsed)")
            part.write_text(json.dumps(rows, indent=2))
            print(f"[{rep} b{b}] wrote {part.name} ({len(rows)} rows, {time.perf_counter()-t_block:.0f}s)")
    print("Phase B complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", default=None, help="comma-separated representation shard")
    ap.add_argument("--blocks", default=None, help="comma-separated block shard")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "A":
        phase_a()
    else:
        phase_b(smoke=args.smoke,
                only_reps=args.reps.split(",") if args.reps else None,
                only_blocks=[int(b) for b in args.blocks.split(",")] if args.blocks else None)


if __name__ == "__main__":
    main()
