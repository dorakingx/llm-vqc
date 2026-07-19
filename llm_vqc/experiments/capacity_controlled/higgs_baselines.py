"""HIGGS classical baselines (evaluated on each block's train-only PCA(8)
features — the SAME controlled representation the VQC receives).

Selection metric = validation log-loss (the primary metric); protected test scored
once after selection. Calibrated SVMs use train-internal CV only (never val/test
labels). Reports test log-loss (primary) + AUROC/accuracy/balanced-accuracy/Brier.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from llm_vqc.evaluation.metrics import logloss
from llm_vqc.evaluation.seeds import TrainingSeeds, derive_child_seed
from llm_vqc.experiments.capacity_controlled import higgs_task as HT
from llm_vqc.experiments.capacity_controlled import t2_task as T2  # full_classification_metrics
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear

warnings.filterwarnings("ignore")
_ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def _test_metrics(pt, yte):
    return T2.full_classification_metrics(pt, yte)


def _sk_grid(build, grid, tv, test):
    Xtr, ytr, Xva, yva = tv.train.features, tv.train.targets, tv.val.features, tv.val.targets
    best = None
    for cfg in grid:
        clf = build(cfg).fit(Xtr, ytr)
        pv = np.clip(clf.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7)
        v = logloss(pv, yva)
        if best is None or v < best["v"]:
            best = {"cfg": cfg, "v": v, "clf": clf}
    pt = np.clip(best["clf"].predict_proba(test.features)[:, 1], 1e-7, 1 - 1e-7)
    return best, pt


def c_majority(tv, test):
    prior = float(np.mean(tv.train.targets))
    pt = np.full(len(test.targets), prior)
    return {"baseline": "C0_majority", "total_params": 0, "val_logloss": None, **_test_metrics(pt, test.targets)}


def c_logreg(tv, test):
    best, pt = _sk_grid(lambda c: LogisticRegression(C=c["C"], max_iter=1000),
                        [{"C": c} for c in (0.1, 1, 10)], tv, test)
    return {"baseline": "C1_logreg", "total_params": tv.train.features.shape[1] + 1,
            "val_logloss": best["v"], "selected": best["cfg"], **_test_metrics(pt, test.targets)}


def c_linsvm(tv, test):
    best, pt = _sk_grid(lambda c: CalibratedClassifierCV(SVC(C=c["C"], kernel="linear"), cv=3),
                        [{"C": c} for c in (0.1, 1, 10)], tv, test)
    return {"baseline": "C2_linear_svm_cal", "total_params": None, "val_logloss": best["v"],
            "selected": best["cfg"], **_test_metrics(pt, test.targets)}


def c_rbfsvm(tv, test):
    best, pt = _sk_grid(lambda c: CalibratedClassifierCV(SVC(C=c["C"], gamma=c["g"], kernel="rbf"), cv=3),
                        [{"C": c, "g": g} for c in (1, 10) for g in ("scale", 0.1)], tv, test)
    return {"baseline": "C3_rbf_svm_cal", "total_params": None, "val_logloss": best["v"],
            "selected": best["cfg"], **_test_metrics(pt, test.targets)}


def c_histgb(tv, test):
    best, pt = _sk_grid(lambda c: HistGradientBoostingClassifier(max_depth=c["d"], learning_rate=c["lr"], random_state=0),
                        [{"d": d, "lr": lr} for d in (3, None) for lr in (0.05, 0.1)], tv, test)
    return {"baseline": "C4_hist_gb", "total_params": None, "val_logloss": best["v"],
            "selected": best["cfg"], **_test_metrics(pt, test.targets)}


def c_rf(tv, test):
    best, pt = _sk_grid(lambda c: RandomForestClassifier(n_estimators=c["n"], max_depth=c["d"], random_state=0),
                        [{"n": n, "d": d} for n in (100, 300) for d in (4, 8, None)], tv, test)
    return {"baseline": "C5_random_forest", "total_params": None, "val_logloss": best["v"],
            "selected": best["cfg"], **_test_metrics(pt, test.targets)}


class _MLP(torch.nn.Module):
    def __init__(self, hidden, activation, in_dim):
        super().__init__(); self.activation = activation
        dims = [in_dim, *hidden, 1]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64)
                                           for i in range(len(dims) - 1)])

    def forward(self, x):
        act = _ACT[self.activation]
        for i, layer in enumerate(self.layers):
            x = layer(x); x = torch.sigmoid(x) if i == len(self.layers) - 1 else act(x)
        return x

    def apply_init(self, seed):
        gen = torch.Generator(); gen.manual_seed(int(seed))
        for layer in self.layers:
            _init_linear(layer, gen)

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))


def _train_mlp(model, tv, lr, wd, seed, epochs=20, batch=16):
    s = TrainingSeeds.from_train_seed(seed); model.apply_init(s.param_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 13, 17], gamma=0.5)
    loss_fn = torch.nn.BCELoss()
    Xtr = torch.tensor(tv.train.features, dtype=torch.float64)
    ytr = torch.tensor(tv.train.targets, dtype=torch.float64).reshape(-1, 1)
    rng = np.random.default_rng(s.minibatch); n = len(Xtr)
    for _ in range(epochs):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(tv.val.features, dtype=torch.float64)).numpy().reshape(-1)
    return np.clip(pv, 1e-7, 1 - 1e-7)


def c_fixed_mlp(tv, test, seed):
    d = tv.train.features.shape[1]
    m = _MLP((5,), "tanh", d); ts = derive_child_seed(seed, "C6")
    _train_mlp(m, tv, 0.05, 1e-5, ts)
    with torch.no_grad():
        pt = np.clip(m(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1), 1e-7, 1 - 1e-7)
    return {"baseline": "C6_fixed_mlp", "total_params": m.n_params(), "val_logloss": None, **_test_metrics(pt, test.targets)}


_NAS_HIDDEN = ((4,), (5,), (6,), (4, 3), (5, 2), (6, 3))
_NAS_ACT = ("relu", "tanh", "sigmoid"); _NAS_LR = (0.01, 0.05, 0.1); _NAS_WD = (0.0, 1e-5, 1e-4)


def c_fair_nas(tv, test, seed, budget=25):
    rng = np.random.default_rng(derive_child_seed(seed, "C7_nas")); d = tv.train.features.shape[1]
    best = None; cand = []
    for i in range(budget):
        cfg = {"hidden": tuple(_NAS_HIDDEN[rng.integers(len(_NAS_HIDDEN))]),
               "activation": _NAS_ACT[rng.integers(len(_NAS_ACT))],
               "lr": float(_NAS_LR[rng.integers(len(_NAS_LR))]), "weight_decay": float(_NAS_WD[rng.integers(len(_NAS_WD))])}
        m = _MLP(cfg["hidden"], cfg["activation"], d); ts = derive_child_seed(seed, "C7_cand", str(i))
        pv = _train_mlp(m, tv, cfg["lr"], cfg["weight_decay"], ts); v = logloss(pv, tv.val.targets)
        cand.append({**cfg, "hidden": list(cfg["hidden"]), "total_params": m.n_params(), "val_logloss": v})
        if best is None or v < best["val_logloss"]:
            with torch.no_grad():
                pt = np.clip(m(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1), 1e-7, 1 - 1e-7)
            best = {**cand[-1], "pt": pt}
    return {"baseline": "C7_fair_nas", "total_params": best["total_params"], "val_logloss": best["val_logloss"],
            "selected": {k: best[k] for k in ("hidden", "activation", "lr", "weight_decay")},
            "n_candidates": len(cand), "candidates": cand, **_test_metrics(best["pt"], test.targets)}


def run_all_classical(tv, test, seed):
    return [c_majority(tv, test), c_logreg(tv, test), c_linsvm(tv, test), c_rbfsvm(tv, test),
            c_histgb(tv, test), c_rf(tv, test), c_fixed_mlp(tv, test, seed), c_fair_nas(tv, test, seed)]
