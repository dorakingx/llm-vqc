#!/usr/bin/env python
"""Development-only task-qualification gate (classical models, NO VQC).

Rejects a saturated or broken task before any quantum benchmark. Fits
preprocessing on dev-training only, evaluates classical families on a held-out
dev-validation split of the qualification region (disjoint from every benchmark
block and from the external holdout), and checks the pre-registered pass criteria.
If HIGGS low-level fails, tries SUSY low-level with the same criteria. Writes
TASK_QUALIFICATION.md / task_qualification.json / task_qualification_results.csv.

No VQC is run here; no subset is chosen because a VQC performs well.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_higgs_v1"
QUAL_SUBSET = 20_000       # rows used for qualification (keeps RBF-SVM tractable)

CRITERIA = {"auroc_min": 0.65, "auroc_max": 0.92, "min_classification_error": 0.05,
            "min_val_logloss": 0.10, "balance_lo": 0.40, "balance_hi": 0.60,
            "min_distinct_families": 2}


def _models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "linear_svm": make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, random_state=0)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0)),
        "hist_gb": HistGradientBoostingClassifier(random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=0)),
    }


def _evaluate(train_X, train_y, val_X, val_y):
    rows = []
    for name, model in _models().items():
        model.fit(train_X, train_y)
        p = model.predict_proba(val_X)[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append({"model": name, "val_auroc": float(roc_auc_score(val_y, p)),
                     "val_logloss": float(log_loss(val_y, p)),
                     "val_classification_error": float(np.mean((p >= 0.5) != val_y))})
    return rows


def _duplicate_across_boundary(train_X, val_X) -> int:
    tr = {hash(r.tobytes()) for r in np.ascontiguousarray(train_X)}
    return sum(1 for r in np.ascontiguousarray(val_X) if hash(r.tobytes()) in tr)


def _check(rows, balance, dup):
    aurocs = [r["val_auroc"] for r in rows]
    best = max(aurocs)
    min_err = min(r["val_classification_error"] for r in rows)
    min_ll = min(r["val_logloss"] for r in rows)
    distinct = len({round(r["val_logloss"], 4) for r in rows})
    checks = {
        "best_auroc_in_range": CRITERIA["auroc_min"] <= best <= CRITERIA["auroc_max"],
        "not_saturated_error": min_err >= CRITERIA["min_classification_error"],
        "val_logloss_above_zero": min_ll >= CRITERIA["min_val_logloss"],
        "class_balance_ok": CRITERIA["balance_lo"] <= balance <= CRITERIA["balance_hi"],
        "at_least_two_families_distinct": distinct >= CRITERIA["min_distinct_families"],
        "no_duplicate_across_boundary": dup == 0,
    }
    return all(checks.values()), checks, {"best_auroc": best, "min_error": min_err,
                                          "min_logloss": min_ll, "balance": balance, "dup": dup}


def _qualify_higgs():
    from llm_vqc.experiments.capacity_controlled import higgs_task as HT
    q = HT.qualification_split(seed=0)
    trX, trY = q["train_X"][:int(QUAL_SUBSET * 0.75)], q["train_y"][:int(QUAL_SUBSET * 0.75)]
    vaX, vaY = q["val_X"][:int(QUAL_SUBSET * 0.25)], q["val_y"][:int(QUAL_SUBSET * 0.25)]
    balance = float(np.mean(np.concatenate([trY, vaY])))
    rows = _evaluate(trX, trY, vaX, vaY)
    dup = _duplicate_across_boundary(trX, vaX)
    passed, checks, summ = _check(rows, balance, dup)
    return {"dataset": "HIGGS_low_level", "n_features": 21, "passed": passed,
            "checks": checks, "summary": summ, "models": rows}


def _qualify_susy():
    from llm_vqc.experiments.capacity_controlled import susy_data as SD
    d = SD.load_dev_pool()  # 8 low-level features
    n = min(QUAL_SUBSET, len(d["features"]))
    X, y = d["features"][:n], d["labels"][:n]
    rng = np.random.default_rng(0); perm = rng.permutation(n); nv = int(n * 0.25)
    vaX, vaY = X[perm[:nv]], y[perm[:nv]]; trX, trY = X[perm[nv:]], y[perm[nv:]]
    balance = float(np.mean(y))
    rows = _evaluate(trX, trY, vaX, vaY)
    dup = _duplicate_across_boundary(trX, vaX)
    passed, checks, summ = _check(rows, balance, dup)
    return {"dataset": "SUSY_low_level", "n_features": 8, "passed": passed,
            "checks": checks, "summary": summ, "models": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"criteria": CRITERIA, "attempts": []}
    higgs = _qualify_higgs()
    result["attempts"].append(higgs)
    selected = None
    if higgs["passed"]:
        selected = "HIGGS_low_level"; result["decision"] = "D1"
    else:
        try:
            susy = _qualify_susy(); result["attempts"].append(susy)
            if susy["passed"]:
                selected = "SUSY_low_level"; result["decision"] = "D2"
            else:
                result["decision"] = "D3"
        except Exception as exc:  # noqa: BLE001
            result["susy_error"] = str(exc); result["decision"] = "D3"
    result["selected_dataset"] = selected

    (OUT / "task_qualification.json").write_text(json.dumps(result, indent=2))
    with (OUT / "task_qualification_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "val_auroc", "val_logloss", "val_classification_error"])
        w.writeheader()
        for att in result["attempts"]:
            for m in att["models"]:
                w.writerow({"dataset": att["dataset"], **m})
    print(f"decision={result['decision']} selected={selected}")
    for att in result["attempts"]:
        print(f"  {att['dataset']}: passed={att['passed']} best_auroc={att['summary']['best_auroc']:.4f} "
              f"min_ll={att['summary']['min_logloss']:.4f} balance={att['summary']['balance']:.3f} checks={att['checks']}")
    return result


if __name__ == "__main__":
    main()
