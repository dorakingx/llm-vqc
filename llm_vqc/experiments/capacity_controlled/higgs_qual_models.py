"""Model families M0-M6 and the configurable VQC training protocol for the HIGGS
data-scale qualification study.

Training operates directly on arrays (not TrainValData) because the qualification
grid varies lr/epochs/weight-decay/early-stopping. Batch size is fixed at 16.
Early stopping (when enabled) uses validation log-loss with patience and restores
the best validation checkpoint. Test labels are never used during training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import TrainingSeeds, derive_child_seed
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear, apply_explicit_init

BATCH = 16
_CLIP = 1e-7
VQC_ARCH_SEEDS = (5000, 5001, 5002, 5003, 5004)


@dataclass(frozen=True)
class Protocol:
    lr: float = 0.05
    epochs: int = 20
    weight_decay: float = 1e-5
    early_stopping: bool = False
    patience: int = 8
    lr_decay_epochs: tuple[int, ...] = (7, 13, 17)
    lr_decay_factor: float = 0.5

    def label(self) -> str:
        es = f"es{self.patience}" if self.early_stopping else "noes"
        return f"lr{self.lr}_ep{self.epochs}_wd{self.weight_decay}_{es}"


# --- metrics ---------------------------------------------------------------

def metrics(pred, y) -> dict:
    p = np.clip(np.asarray(pred, float).reshape(-1), _CLIP, 1 - _CLIP)
    yy = np.asarray(y, float).reshape(-1)
    yhat = (p >= 0.5).astype(float)
    ll = float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
    try:
        auc = float(roc_auc_score(yy, p)) if len(set(yy.tolist())) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        auc = float("nan")
    tpr = float(((yhat == 1) & (yy == 1)).sum() / max(1, (yy == 1).sum()))
    tnr = float(((yhat == 0) & (yy == 0)).sum() / max(1, (yy == 0).sum()))
    # 10-bin expected calibration error
    bins = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - yy[m].mean())
    return {"logloss": ll, "auroc": auc, "balanced_accuracy": 0.5 * (tpr + tnr),
            "classification_error": float(np.mean(yhat != yy)),
            "brier": float(np.mean((p - yy) ** 2)), "ece": float(ece)}


# --- M0/M1/M2 classical ----------------------------------------------------

def m0_class_prior(Xtr, ytr, Xva, Xte):
    prior = float(np.mean(ytr))
    return np.full(len(Xva), prior), np.full(len(Xte), prior), 0


def m1_logreg(Xtr, ytr, Xva, yva, Xte):
    best = None
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=1000).fit(Xtr, ytr)
        pv = np.clip(clf.predict_proba(Xva)[:, 1], _CLIP, 1 - _CLIP)
        ll = metrics(pv, yva)["logloss"]
        if best is None or ll < best[0]:
            best = (ll, clf, pv)
    _, clf, pv = best
    return pv, np.clip(clf.predict_proba(Xte)[:, 1], _CLIP, 1 - _CLIP), Xtr.shape[1] + 1


class _MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = torch.nn.Linear(dim, 4, dtype=torch.float64)
        self.l2 = torch.nn.Linear(4, 1, dtype=torch.float64)

    def forward(self, x):
        return torch.sigmoid(self.l2(torch.tanh(self.l1(x))))

    def apply_init(self, seed):
        g = torch.Generator(); g.manual_seed(int(seed))
        _init_linear(self.l1, g); _init_linear(self.l2, g)


def _train_torch(model, Xtr, ytr, Xva, yva, proto: Protocol, seed: int, extra_frozen=None):
    """Shared loop: AdamW + MultiStepLR + BCE, optional early stopping on val log-loss."""
    s = TrainingSeeds.from_train_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=proto.lr, weight_decay=proto.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(proto.lr_decay_epochs),
                                                 gamma=proto.lr_decay_factor)
    lf = torch.nn.BCELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float64); yt = torch.tensor(ytr, dtype=torch.float64).reshape(-1, 1)
    Xv = torch.tensor(Xva, dtype=torch.float64)
    rng = np.random.default_rng(s.minibatch); n = len(Xt)
    best_ll, best_state, bad, epochs_run = float("inf"), None, 0, 0
    for ep in range(proto.epochs):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lf(model(Xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                return {"failed": True, "epochs_run": ep}
            loss.backward(); opt.step()
        sched.step(); epochs_run = ep + 1
        if not proto.early_stopping:
            # per-epoch validation is only needed to drive early stopping /
            # best-checkpoint restore; skipping it here is semantically identical.
            continue
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy().reshape(-1)
        ll = metrics(pv, yva)["logloss"]
        if ll < best_ll - 1e-6:
            best_ll, bad = ll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if proto.early_stopping and bad >= proto.patience:
                break
    if proto.early_stopping and best_state is not None:
        model.load_state_dict(best_state)   # restore best validation checkpoint
    return {"failed": False, "epochs_run": epochs_run, "best_val_logloss": best_ll}


def m2_mlp(Xtr, ytr, Xva, yva, Xte, proto, seed):
    m = _MLP(Xtr.shape[1]); m.apply_init(TrainingSeeds.from_train_seed(seed).param_init)
    r = _train_torch(m, Xtr, ytr, Xva, yva, proto, seed)
    m.eval()
    with torch.no_grad():
        pv = m(torch.tensor(Xva, dtype=torch.float64)).numpy().reshape(-1)
        pt = m(torch.tensor(Xte, dtype=torch.float64)).numpy().reshape(-1)
    n_params = int(sum(p.numel() for p in m.parameters()))
    return pv, pt, n_params, r


# --- M3-M6 VQC -------------------------------------------------------------

def product_arch(seed: int):
    """Fixed product-state architecture (no two-qubit gates): 3 single-qubit
    rotation blocks drawn deterministically from the predeclared seed."""
    rng = np.random.default_rng(seed)
    kinds = [str(rng.choice(["RX", "RY", "RZ"])) for _ in range(S.N_PARAM_BLOCKS)]
    return S.ControlledGenome(blocks=[S.ParamBlock(kind=k) for k in kinds])


def entangled_arch(seed: int):
    """Fixed entangled architecture: >=1 CRZ param block, no free entangling
    blocks (so its exact CRZ->RZ product counterpart is entanglement-free)."""
    rng = np.random.default_rng(seed)
    while True:
        g = S.random_genome(rng)
        has_crz = any(b.role == "param" and b.kind == "CRZ" for b in g.blocks)
        free_ent = any(b.role == "free" and S.FREE_BLOCK_TEMPLATES[b.template_index][0] == "entangle"
                       for b in g.blocks)
        if has_crz and not free_ent:
            return g


def train_vqc(ir, dim, Xtr, ytr, Xva, yva, Xte, proto: Protocol, seed: int, freeze_quantum=False):
    t0 = time.perf_counter()
    s = TrainingSeeds.from_train_seed(seed)
    torch.manual_seed(s.param_init)
    model = HybridQNNModel(ir, raw_feature_dim=dim, head_out_dim=1)
    apply_explicit_init(model, s.param_init)
    q_init = model.q_layer.weights.detach().clone()
    if freeze_quantum:
        model.q_layer.weights.requires_grad_(False)
    r = _train_torch(model, Xtr, ytr, Xva, yva, proto, seed)
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xva, dtype=torch.float64)).numpy().reshape(-1)
        pt = model(torch.tensor(Xte, dtype=torch.float64)).numpy().reshape(-1)
    named = dict(model.named_parameters())
    cap = {"embed_params": int(sum(p.numel() for n, p in named.items() if n.startswith("embed."))),
           "quantum_params": int(sum(p.numel() for n, p in named.items() if n.startswith("q_layer."))),
           "head_params": int(sum(p.numel() for n, p in named.items() if n.startswith("head."))),
           "frozen_quantum_params": 12 if freeze_quantum else 0}
    cap["total_params"] = cap["embed_params"] + cap["quantum_params"] + cap["head_params"]
    cap["total_trainable_params"] = cap["total_params"] - cap["frozen_quantum_params"]
    return {"val_pred": pv, "test_pred": pt, "capacity": cap, "runtime_s": time.perf_counter() - t0,
            "q_unchanged": bool(torch.allclose(q_init, model.q_layer.weights.detach())), **r}
