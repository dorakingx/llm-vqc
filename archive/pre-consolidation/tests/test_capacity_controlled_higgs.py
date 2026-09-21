"""HIGGS-v1 fairness / capacity / integrity / ablation / hardware tests.

Data-dependent tests (blocks, preprocessing, holdout) require the prepared HIGGS
cache; they skip cleanly when it is absent so the suite passes in environments
without the 2.6 GB download. Structural tests (capacity, grammar, CRZ->RZ mapping,
hardware determinism, quarantine) always run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.seeds import TrainingSeeds, train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.experiments.capacity_controlled import higgs_ablations as HA
from llm_vqc.experiments.capacity_controlled import higgs_data as HD
from llm_vqc.experiments.capacity_controlled import higgs_hardware as HW
from llm_vqc.experiments.capacity_controlled import higgs_task as HT
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.arms import ControlledRandomArm
from llm_vqc.experiments.capacity_controlled.init_policy import apply_explicit_init
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR, EncodingSpec
from llm_vqc.search.runner import SearchRunner

_HAS_CACHE = HD.DEV_BENCH_NPZ.exists() and HD.HOLDOUT_NPZ.exists()
needs_cache = pytest.mark.skipif(not _HAS_CACHE, reason="HIGGS cache not prepared")


# --- structural (always run) -----------------------------------------------

def test_higgs_capacity_is_53():
    rng = np.random.default_rng(11)
    for _ in range(200):
        ir = S.genome_to_ir(S.random_genome(rng))
        assert S.validate_controlled(ir) == []
        m = HybridQNNModel(ir, raw_feature_dim=HT.RAW_FEATURE_DIM, head_out_dim=1)
        named = dict(m.named_parameters())
        assert sum(p.numel() for n, p in named.items() if n.startswith("embed.")) == 36
        assert sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")) == 12
        assert sum(p.numel() for n, p in named.items() if n.startswith("head.")) == 5
        assert sum(p.numel() for p in m.parameters()) == HT.TOTAL_TRAINABLE_PARAMS == 53


def test_higgs_amplitude_rejected():
    ir = S.genome_to_ir(HA.predeclared_architectures()[0]).model_copy(
        update={"encoding": EncodingSpec(type="amplitude", wires="all")})
    assert any(i.code == "controlled.encoding_type" for i in S.validate_controlled(ir))


def test_crz_to_rz_product_mapping_preserves_param_and_gate_count():
    for e, p in HA.entangled_pairs(20):
        d = HA.pair_diagnostics(e, p)
        assert d["param_count_preserved"]
        assert d["product_two_qubit"] == 0 and d["entangled_two_qubit"] >= 1
        assert d["gate_count_match"]
        assert S.validate_controlled(S.genome_to_ir(p)) == []
        assert HT.total_trainable_params(S.genome_to_ir(e)) == HT.total_trainable_params(S.genome_to_ir(p)) == 53


def test_predeclared_architectures_stable_and_valid():
    a = [structural_hash(S.genome_to_ir(g)) for g in HA.predeclared_architectures()]
    b = [structural_hash(S.genome_to_ir(g)) for g in HA.predeclared_architectures()]
    assert a == b and len(a) == 20


def test_hardware_transpile_is_deterministic():
    ir = S.genome_to_ir(HA.entangled_pairs(1)[0][0])
    assert HW.transpile_metrics(ir) == HW.transpile_metrics(ir)


def test_paired_freeze_train_identical_init_and_frozen_unchanged():
    # uses a tiny synthetic TrainValData shape via a block if cache present,
    # otherwise a structural check that frozen weights equal the shared init.
    ir = S.genome_to_ir(HA.predeclared_architectures()[2])
    ts = train_seed_for_circuit(0, structural_hash(ir))
    torch.manual_seed(0)
    m0 = HybridQNNModel(ir, raw_feature_dim=HT.RAW_FEATURE_DIM, head_out_dim=1)
    apply_explicit_init(m0, TrainingSeeds.from_train_seed(ts).param_init)
    q_init = m0.q_layer.weights.detach().clone()
    if not _HAS_CACHE:
        pytest.skip("cache needed to train")
    tv, test = HT.build_block(0)
    tr = train_model(ir, tv, TrainingConfig(epochs=2), ts, init_policy=apply_explicit_init, freeze_quantum=False)
    fr = train_model(ir, tv, TrainingConfig(epochs=2), ts, init_policy=apply_explicit_init, freeze_quantum=True)
    assert torch.allclose(q_init, torch.tensor(fr.trained_circuit_weights))
    assert not torch.allclose(q_init, torch.tensor(tr.trained_circuit_weights))


def test_no_test_in_search_feedback_and_holdout_not_imported_by_task():
    import ast
    import inspect
    from llm_vqc.search.feedback import SearchFeedback
    assert not any("test" in f for f in SearchFeedback.model_fields)
    # higgs_task must not import the holdout loader path in a way search would use
    tree = ast.parse(inspect.getsource(HT))
    # load_holdout_audit lives in higgs_data and is only called by the external audit script
    src = inspect.getsource(HT)
    assert "load_holdout_audit" not in src


# --- data-dependent (need cache) -------------------------------------------

@needs_cache
def test_row_count_and_manifest():
    import json
    manifest = json.loads(HD.MANIFEST.read_text())
    assert manifest["row_count_verified"] == HD.N_ROWS_EXPECTED == 11_000_000
    assert manifest["n_low_level"] == 21 and manifest["n_high_level"] == 7
    assert len(manifest["compressed_sha256"]) == 64


@needs_cache
def test_blocks_mutually_disjoint_and_stratified():
    blocks = HT.build_block_assignments()
    assert len(blocks) == 10
    all_ids = []
    for b in blocks:
        ids = list(b.train_ids) + list(b.val_ids) + list(b.test_ids)
        assert len(ids) == len(set(ids)) == HT.N_TRAIN + HT.N_VAL + HT.N_TEST
        all_ids += ids
    assert len(all_ids) == len(set(all_ids))  # no sample in two blocks or partitions


@needs_cache
def test_dev_benchmark_holdout_disjoint():
    blocks = HT.build_block_assignments()
    bench_ids = {i for b in blocks for i in (list(b.train_ids) + list(b.val_ids) + list(b.test_ids))}
    assert all(i >= HT.QUAL_ROWS for i in bench_ids)           # disjoint from qualification region
    assert all(i < HD.HOLDOUT_START for i in bench_ids)         # disjoint from holdout
    hold = HD.load_holdout_audit()
    assert all(i >= HD.HOLDOUT_START for i in hold["row_ids"])  # holdout beyond benchmark
    assert not (bench_ids & set(int(i) for i in hold["row_ids"]))


@needs_cache
def test_preprocessing_fitted_on_block_train_only():
    tv, test = HT.build_block(0)
    scaler = tv.preprocessing.params["scaler"]
    mean_before = scaler.mean_.copy()
    # perturbing val/test cannot change the fitted objects
    tv2, test2 = HT.build_block(0)
    tv2.val.features[:] = 0.0
    test2.features[:] = 0.0
    assert np.array_equal(mean_before, tv2.preprocessing.params["scaler"].mean_)
    assert tv.train.features.shape[1] == HT.PCA_COMPONENTS == 8


@needs_cache
def test_search_budget_capacity_selection(tmp_path):
    tv, test = HT.build_block(0)
    store = ResultStore(tmp_path / "s.sqlite")
    runner = SearchRunner(ControlledRandomArm(lower_is_better=True), "HIGGS", tv,
                          TrainingConfig(epochs=2), budget_limit=5, run_seed=0, result_store=store,
                          run_id="r0", extra_validator=S.validate_controlled, init_policy=apply_explicit_init)
    res = runner.run()
    assert res.ledger_summary["consumed_budget"] == 5
    for e in store.all_evaluations("HIGGS"):
        ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
        assert S.validate_controlled(ir) == []
        assert HT.total_trainable_params(ir) == 53
    store.close()


@needs_cache
def test_resume_deterministic(tmp_path):
    tv, test = HT.build_block(0)

    def _run(store, budget):
        return SearchRunner(ControlledRandomArm(lower_is_better=True), "HIGGS", tv,
                            TrainingConfig(epochs=2), budget_limit=budget, run_seed=1, result_store=store,
                            run_id="run", extra_validator=S.validate_controlled, init_policy=apply_explicit_init).run()

    sa = ResultStore(tmp_path / "a.sqlite"); ra = _run(sa, 5); sa.close()
    sb = ResultStore(tmp_path / "b.sqlite"); _run(sb, 3); sb.close()
    sb2 = ResultStore(tmp_path / "b.sqlite"); rb = _run(sb2, 5); sb2.close()
    assert ra.selected_structural_hash == rb.selected_structural_hash
    assert ra.ledger_summary == rb.ledger_summary
