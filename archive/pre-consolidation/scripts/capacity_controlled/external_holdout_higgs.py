#!/usr/bin/env python
"""External-holdout audit — run ONCE, only after the protocol is committed, all
searches complete, and analyses are frozen. Implements the pre-registered
consensus rule, retrains on a preregistered combined training set, and evaluates
one time on the fixed 20,000-row subset of the official final 500,000 HIGGS
holdout. No design decision is changed from the result.

Consensus architecture = the selected structural hash with the lowest MEAN
selected-validation log-loss across the 10 blocks. Combined training set = the 10
blocks' training partitions (5,000 samples); preprocessing (StandardScaler+PCA(8))
fit on that combined training set only; holdout transformed by it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import higgs_data as HD
from llm_vqc.experiments.capacity_controlled import higgs_task as HT
from llm_vqc.experiments.capacity_controlled import t2_task as T2
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import DataSplit, TrainValData

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_higgs_v1"
RUN_DIR = ROOT / "runs" / "capacity_controlled_higgs_v1"
TASK = "HIGGS"


def _consensus_hash(summary):
    by_hash = defaultdict(list)
    for r in summary["runs"]:
        if r["selected_structural_hash"] and r["selected_val_logloss"] is not None:
            by_hash[r["selected_structural_hash"]].append((r["block"], r["seed"], r["selected_val_logloss"]))
    ranked = sorted(by_hash.items(), key=lambda kv: np.mean([x[2] for x in kv[1]]))
    h, occ = ranked[0]
    return h, {"structural_hash": h[:12], "n_selections": len(occ),
               "mean_selected_val_logloss": float(np.mean([x[2] for x in occ])),
               "blocks": sorted({x[0] for x in occ})}


def main():
    summary = json.loads((RUN_DIR / "pilot_summary.json").read_text())
    assignments = HT.build_block_assignments()
    consensus_hash, meta = _consensus_hash(summary)

    # get the consensus IR from any block store
    ir_json = None
    for b in summary["blocks"]:
        st = ResultStore(RUN_DIR / f"block_{b}" / "results.sqlite")
        ts = train_seed_for_circuit(0, consensus_hash)
        c = st.get_cached(TASK, consensus_hash, ts)
        st.close()
        if c and c.circuit_canonical_json:
            ir_json = c.circuit_canonical_json; break
    ir = CircuitIR.model_validate_json(ir_json)
    HT.assert_capacity(ir)

    # combined training set from the 10 blocks' train partitions (raw 8 low-level PCA? no: raw low-level)
    pool = HD.load_dev_benchmark_pool(low_level_only=True)
    id_to_pos = {int(rid): i for i, rid in enumerate(pool["row_ids"])}
    train_ids = [i for a in assignments for i in a.train_ids]
    pos = [id_to_pos[i] for i in train_ids]
    Xtr_raw = pool["features"][pos]; ytr = pool["labels"][pos].astype(np.float64)

    # preprocessing fit on combined train only; transform holdout
    prep = HT._fit_preprocessing(Xtr_raw)
    hold = HD.load_holdout_audit(low_level_only=True)   # FIRST access to holdout
    Xho = prep.transform(hold["features"]); yho = hold["labels"].astype(np.float64)
    Xtr = prep.transform(Xtr_raw)

    # retrain consensus architecture on combined training set
    spec = HT.SPEC
    train = DataSplit(features=Xtr, targets=ytr, sample_ids=tuple(f"comb-{i}" for i in train_ids))
    # a small internal val (not the holdout) just to satisfy the training loop's val metric
    n_val = 500
    val = DataSplit(features=Xtr[:n_val], targets=ytr[:n_val], sample_ids=tuple(f"combv-{i}" for i in train_ids[:n_val]))
    tv = TrainValData(spec=spec, train=train, val=val, split_seed=99, preprocessing=prep)
    out = train_model(ir, tv, TrainingConfig(), train_seed=train_seed_for_circuit(0, consensus_hash),
                      init_policy=apply_explicit_init)

    model = HybridQNNModel(ir, raw_feature_dim=HT.RAW_FEATURE_DIM, head_out_dim=1)
    sd = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
          for k, v in out.trained_classical_state.items()}
    model.load_state_dict(sd); model.eval()
    with torch.no_grad():
        pho = model(torch.tensor(Xho, dtype=torch.float64)).numpy().reshape(-1)
    md = T2.full_classification_metrics(pho, yho, with_curves=False)

    result = {"note": "External-holdout audit, evaluated ONCE after design freeze. No design change made.",
              "consensus_rule": "selected structural hash with lowest mean selected-validation log-loss across 10 blocks",
              "consensus_architecture": meta,
              "combined_training_samples": len(train_ids),
              "holdout_audit_rows": len(yho), "holdout_range": HD.HOLDOUT_NPZ.name,
              "external_holdout_metrics": {k: md[k] for k in ("logloss", "auc", "accuracy", "balanced_accuracy_error", "brier")},
              "confusion": md["confusion"]}
    (OUT / "external_holdout_result.json").write_text(json.dumps(result, indent=2))

    # fig22 external audit
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["log-loss", "1-AUROC", "class-error"], [md["logloss"], 1 - md["auc"], md["classification_error"]], color="#3767b3", alpha=0.8)
    ax.set_title(f"External-holdout audit (20k official test)\nconsensus arch {meta['structural_hash']}, AUROC={md['auc']:.3f}", fontsize=9)
    fig.savefig(OUT / "fig23_external_holdout.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig23_external_holdout.svg", bbox_inches="tight"); plt.close(fig)

    print(f"consensus={meta['structural_hash']} holdout AUROC={md['auc']:.4f} logloss={md['logloss']:.4f} acc={md['accuracy']:.4f}")
    print(f"wrote {OUT}/external_holdout_result.json")


if __name__ == "__main__":
    main()
