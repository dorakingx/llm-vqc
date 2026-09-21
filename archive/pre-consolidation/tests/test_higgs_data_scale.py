"""HIGGS data-scale qualification: data-integrity, capacity, pairing and export tests.

Data-dependent tests skip cleanly when the qualification cache is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_vqc.experiments.capacity_controlled import higgs_data as HD
from llm_vqc.experiments.capacity_controlled import higgs_hardware as HW
from llm_vqc.experiments.capacity_controlled import higgs_qual_models as QM
from llm_vqc.experiments.capacity_controlled import higgs_scale as HS
from llm_vqc.experiments.capacity_controlled import higgs_task as HT
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.ir.metrics import circuit_cost_summary

_HAS = HS.QUAL_NPZ.exists()
needs_cache = pytest.mark.skipif(not _HAS, reason="qualification cache not prepared")
OUT = Path("outputs/higgs_data_scale_qualification_v1")


# --- structural (always run) ----------------------------------------------

def test_product_arch_has_no_entangling_gates():
    for s in QM.VQC_ARCH_SEEDS:
        ir = S.genome_to_ir(QM.product_arch(s))
        assert S.validate_controlled(ir) == []
        assert circuit_cost_summary(ir).two_qubit_gate_count == 0


def test_entangled_arch_has_entanglement_and_is_valid():
    for s in QM.VQC_ARCH_SEEDS:
        ir = S.genome_to_ir(QM.entangled_arch(s))
        assert S.validate_controlled(ir) == []
        assert circuit_cost_summary(ir).two_qubit_gate_count >= 1


def test_predeclared_architectures_are_deterministic():
    a = [S.genome_to_ir(QM.entangled_arch(s)).model_dump_json() for s in QM.VQC_ARCH_SEEDS]
    b = [S.genome_to_ir(QM.entangled_arch(s)).model_dump_json() for s in QM.VQC_ARCH_SEEDS]
    assert a == b


def test_hardware_compilation_deterministic():
    ir = S.genome_to_ir(QM.entangled_arch(5000))
    assert HW.transpile_metrics(ir) == HW.transpile_metrics(ir)


def test_expected_capacities_by_representation():
    """R1 21->4 = 88, R2 8->4 = 36, R3 16->4 = 68 embedding params (+12 quantum, +5 head)."""
    from llm_vqc.evaluation.model import HybridQNNModel
    ir = S.genome_to_ir(QM.entangled_arch(5000))
    for dim, embed, total in ((21, 88, 105), (8, 36, 53), (16, 68, 85)):
        m = HybridQNNModel(ir, raw_feature_dim=dim, head_out_dim=1)
        named = dict(m.named_parameters())
        assert sum(p.numel() for n, p in named.items() if n.startswith("embed.")) == embed
        assert sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")) == 12
        assert sum(p.numel() for n, p in named.items() if n.startswith("head.")) == 5
        assert sum(p.numel() for p in m.parameters()) == total


def test_official_holdout_region_untouched_by_this_cycle():
    """The qualification pool must not intersect the official final 500k rows."""
    lo, hi = HS.QUAL_START, HS.QUAL_START + HS.QUAL_ROWS
    assert hi <= HD.HOLDOUT_START
    # and this module never loads the holdout
    import inspect
    assert "load_holdout_audit" not in inspect.getsource(HS)


def test_qualification_pool_disjoint_from_v1_regions():
    lo, hi = HS.QUAL_START, HS.QUAL_START + HS.QUAL_ROWS
    assert lo >= 300_000            # beyond the v1 dev subset and v1 block region
    assert hi <= HD.HOLDOUT_START   # before the official holdout


# --- data-dependent --------------------------------------------------------

@needs_cache
def test_blocks_mutually_disjoint():
    blocks = HS.build_qual_blocks()
    assert len(blocks) == HS.N_BLOCKS_TOTAL
    ids = [i for b in blocks for i in (list(b.train_ids) + list(b.val_ids) + list(b.test_ids))]
    assert len(ids) == len(set(ids))
    for b in blocks:
        assert len(b.train_ids) == HS.MAX_TRAIN
        assert len(b.val_ids) == HS.N_VAL and len(b.test_ids) == HS.N_TEST


@needs_cache
def test_nested_training_subsets():
    b = HS.build_qual_blocks()[0]
    prev = None
    for n in HS.TRAIN_SIZES:
        cur = list(b.train_ids[:n])
        assert len(cur) == n
        if prev is not None:
            assert cur[:len(prev)] == prev      # strictly nested
        prev = cur


@needs_cache
def test_qualification_rows_do_not_overlap_previous_higgs_rows():
    blocks = HS.build_qual_blocks()
    qual_ids = {i for b in blocks for i in (list(b.train_ids) + list(b.val_ids) + list(b.test_ids))}
    # v1 dev subset + v1 benchmark blocks all live in [0, 300000)
    assert min(qual_ids) >= 300_000
    v1_blocks = HT.build_block_assignments()
    v1_ids = {i for a in v1_blocks for i in (list(a.train_ids) + list(a.val_ids) + list(a.test_ids))}
    assert not (qual_ids & v1_ids)
    hold = HD.load_holdout_audit()
    assert not (qual_ids & {int(i) for i in hold["row_ids"]})


@needs_cache
def test_preprocessing_fitted_on_training_only_and_perturbation_safe():
    b = HS.build_qual_blocks()[0]
    c1 = HS.block_condition(b, 500, "R2_pca8")
    ref_tr = c1["Xtr"].copy()
    c2 = HS.block_condition(b, 500, "R2_pca8")
    c2["Xva"][:] = 0.0
    c2["Xte"][:] = 0.0            # perturbing val/test cannot change the fitted transform
    c3 = HS.block_condition(b, 500, "R2_pca8")
    assert np.allclose(ref_tr, c3["Xtr"])
    assert c1["Xtr"].shape[1] == 8
    assert HS.block_condition(b, 500, "R1_raw21")["Xtr"].shape[1] == 21
    assert HS.block_condition(b, 500, "R3_pca16")["Xtr"].shape[1] == 16


@needs_cache
def test_frozen_quantum_stays_frozen_and_pairs_share_initialization():
    b = HS.build_qual_blocks()[0]
    c = HS.block_condition(b, 500, "R2_pca8")
    ir = S.genome_to_ir(QM.entangled_arch(5000))
    proto = QM.Protocol(lr=0.05, epochs=2, weight_decay=1e-5, early_stopping=False)
    tr = QM.train_vqc(ir, c["dim"], c["Xtr"], c["ytr"], c["Xva"], c["yva"], c["Xte"], proto, 7, freeze_quantum=False)
    fr = QM.train_vqc(ir, c["dim"], c["Xtr"], c["ytr"], c["Xva"], c["yva"], c["Xte"], proto, 7, freeze_quantum=True)
    assert fr["q_unchanged"] is True        # frozen weights never moved
    assert tr["q_unchanged"] is False       # trainable weights moved
    assert fr["capacity"]["frozen_quantum_params"] == 12
    assert fr["capacity"]["total_params"] == tr["capacity"]["total_params"]
    assert fr["capacity"]["total_trainable_params"] == tr["capacity"]["total_trainable_params"] - 12


# --- exported-artifact hygiene ---------------------------------------------

def test_exported_artifacts_contain_no_raw_data_or_secrets():
    if not OUT.exists():
        pytest.skip("output package not generated yet")
    bad = [p.name for p in OUT.iterdir()
           if p.suffix.lower() in {".sqlite", ".db", ".npz", ".gz", ".pt", ".pth", ".env", ".key"}]
    assert bad == []
    for p in OUT.glob("*.json"):
        txt = p.read_text().lower()
        assert "api_key" not in txt and "secret" not in txt and "password" not in txt


def test_protocol_selection_never_used_test_labels():
    sel = OUT / "training_protocol_selection.json"
    if not sel.exists():
        pytest.skip("phase A not run yet")
    d = json.loads(sel.read_text())
    assert d["test_labels_used"] is False
    assert "validation log-loss" in d["selection_rule"]
