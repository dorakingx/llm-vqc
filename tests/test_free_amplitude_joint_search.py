"""Tests for the main-mode joint structure-and-theta search (correction
pass "Tests and execution"): exact proposed theta is evaluated; no
optimizer exists in the main mode; samplewise loss has no broadcasting;
Random theta is in range; candidate_hash changes when theta changes;
Open-loop contains no history; Closed-loop contains prior losses and
theta; Test metrics never enter prompts; Expressibility cache is keyed by
architecture_hash; fixed-theta diagnostics are not labelled
expressibility; all required figure and source-data files are generated.

Every provider here is an offline fake -- no network access anywhere.
"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import time

import numpy as np
import pytest
import torch

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.candidate_schema import (
    architecture_hash,
    candidate_hash,
    complete_candidate_to_ir_and_theta,
    validate_complete_candidate,
)
from llm_vqc.free_amplitude.main_eval import MainModeCache, evaluate_complete_candidate
from llm_vqc.free_amplitude.main_runner import (
    run_closed_loop_arm_main,
    run_open_loop_arm_main,
    run_random_arm_main,
)
from llm_vqc.free_amplitude.metrics import (
    SamplewiseShapeError,
    samplewise_mae,
    samplewise_mse,
    samplewise_rmse,
)
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.free_amplitude.sampler import sample_complete_candidate
from llm_vqc.free_amplitude.tasks import AMPLITUDE_N3_SMOKE_V1, AmplitudeGaussianPeakTask
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.llm.provider import LLMResponse

TASK = "joint_test"


@pytest.fixture(scope="module")
def train_val():
    tv, _ = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1).build(seed=0)
    return tv


def _store(tmp_path, name="r.sqlite"):
    return ResultStore.open_or_create(
        tmp_path / name, run_id="run", config_json="{}",
        config_reproducibility_fields={"t": 1}, git_sha=None, created_at="t0",
        allow_incompatible=True,
    )


class _ScriptedProvider:
    model_name = "scripted-joint-test"

    def __init__(self, responses):
        self._responses = responses
        self.calls_made = 0
        self.prompts_seen = []

    def complete(self, system_prompt, user_prompt, temperature):
        self.prompts_seen.append(user_prompt)
        text = self._responses[self.calls_made]
        self.calls_made += 1
        return LLMResponse(
            raw_text=text, model=self.model_name, input_tokens=5, output_tokens=5,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


def _cand(theta_ry: float) -> str:
    return json.dumps({
        "n_qubits": 3,
        "operations": [{"gate": "RY", "wires": [0], "theta": theta_ry}],
    })


# --- exact proposed theta is evaluated --------------------------------------


def test_exact_proposed_theta_is_evaluated(train_val):
    """The model's quantum weights after loading must equal the proposed
    theta verbatim, and the resulting val RMSE must match an independent
    forward pass at exactly those angles."""
    p = {"n_qubits": 3, "operations": [
        {"gate": "H", "wires": [1]},
        {"gate": "CRY", "wires": [1, 0], "theta": 1.247},
        {"gate": "RZ", "wires": [2], "theta": -0.381},
    ]}
    v = validate_complete_candidate(p, 3)
    ir, theta = complete_candidate_to_ir_and_theta(v.proposal, 0)
    assert theta == [1.247, -0.381]

    model = FixedReadoutQuantumModel(ir, readout_qubit=0)
    model.double()
    with torch.no_grad():
        model.q_layer.weights.copy_(torch.tensor(theta, dtype=torch.float64))
    assert model.q_layer.weights.detach().tolist() == [1.247, -0.381]

    ledger = BudgetLedger()
    result = evaluate_complete_candidate(p, 3, 0, train_val, "p0", ledger, seed=0)
    with torch.no_grad():
        preds = model(torch.tensor(train_val.val.features, dtype=torch.float64))
    expected_rmse = samplewise_rmse(preds.numpy(), train_val.val.targets)
    assert result.val_rmse == pytest.approx(expected_rmse, abs=1e-12)


# --- no optimizer exists in the main mode -----------------------------------


def test_no_optimizer_in_main_mode():
    """`main_eval` and `main_runner` must not import torch.optim or
    construct any optimizer -- checked by AST inspection of the source."""
    for module in ("main_eval.py", "main_runner.py"):
        source = pathlib.Path(f"llm_vqc/free_amplitude/{module}").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("AdamW", "SGD", "Adam"):
                raise AssertionError(f"{module} references torch.optim.{node.attr}")
            if isinstance(node, ast.ImportFrom) and "optim" in (node.module or ""):
                raise AssertionError(f"{module} imports {node.module}")
        assert "optimizer" not in source.lower() or "no optimizer" in source.lower()


# --- samplewise loss has no broadcasting ------------------------------------


def test_samplewise_loss_no_broadcasting():
    preds_col = np.random.rand(5, 1)  # (B, 1)
    targets_flat = np.random.rand(5)  # (B,)
    # Flattened alignment: fine, and equals the loop-computed value.
    mse = samplewise_mse(preds_col, targets_flat)
    manual = float(np.mean([(preds_col[i, 0] - targets_flat[i]) ** 2 for i in range(5)]))
    assert mse == pytest.approx(manual)

    # A genuinely mismatched pair must raise, never silently broadcast.
    with pytest.raises(SamplewiseShapeError):
        samplewise_mse(np.random.rand(5), np.random.rand(4))
    with pytest.raises(SamplewiseShapeError):
        samplewise_mae(np.random.rand(6, 1), np.random.rand(5))


# --- Random theta is in range ------------------------------------------------


def test_random_theta_in_range():
    rng = np.random.default_rng(0)
    for _ in range(300):
        c = sample_complete_candidate(rng, 3, 5)
        for op in c.operations:
            if op.theta is not None:
                assert -math.pi <= op.theta <= math.pi


# --- candidate_hash changes when theta changes ------------------------------


def test_candidate_hash_changes_when_theta_changes():
    p1 = validate_complete_candidate(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0], "theta": 0.1}]}, 3
    ).proposal
    p2 = validate_complete_candidate(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0], "theta": 0.2}]}, 3
    ).proposal
    assert architecture_hash(p1, 0) == architecture_hash(p2, 0)
    assert candidate_hash(p1, 0) != candidate_hash(p2, 0)


def test_same_structure_different_theta_is_not_duplicate(tmp_path, train_val):
    store = _store(tmp_path)
    cache = MainModeCache(store, TASK, seed=0)
    ledger = BudgetLedger()
    r1 = evaluate_complete_candidate(
        json.loads(_cand(0.1)), 3, 0, train_val, "p0", ledger, seed=0, cache=cache
    )
    r2 = evaluate_complete_candidate(
        json.loads(_cand(0.9)), 3, 0, train_val, "p1", ledger, seed=0, cache=cache
    )
    r3 = evaluate_complete_candidate(
        json.loads(_cand(0.1)), 3, 0, train_val, "p2", ledger, seed=0, cache=cache
    )
    store.close()
    assert not r1.is_duplicate
    assert not r2.is_duplicate  # different theta -> different candidate
    assert r3.is_duplicate  # identical structure+theta -> duplicate


# --- Open-loop contains no history ------------------------------------------


def test_open_loop_contains_no_history(tmp_path, train_val):
    store = _store(tmp_path)
    provider = _ScriptedProvider([_cand(0.3), _cand(0.7)])
    run_open_loop_arm_main(
        0, TASK, 3, 0, train_val, 2, store, provider, time.monotonic() + 60, feature_count=8,
    )
    store.close()
    assert len(provider.prompts_seen) == 2
    assert provider.prompts_seen[0] == provider.prompts_seen[1]
    for prompt in provider.prompts_seen:
        assert "history" not in prompt.lower()


# --- Closed-loop contains prior losses and theta ----------------------------


def test_closed_loop_contains_prior_losses_and_theta(tmp_path, train_val):
    store = _store(tmp_path)
    provider = _ScriptedProvider([_cand(0.3), _cand(0.7), _cand(-1.1)])
    run_closed_loop_arm_main(
        0, TASK, 3, 0, train_val, 3, store, provider, time.monotonic() + 60, feature_count=8,
    )
    store.close()
    assert len(provider.prompts_seen) == 3
    first, second, third = provider.prompts_seen
    assert "{" not in first  # no feedback block on the very first turn

    fb2 = json.loads(second[second.index("{"): second.rindex("}") + 1])
    assert len(fb2["history"]) == 1
    entry = fb2["history"][0]
    assert entry["theta"] == [0.3]
    assert entry["train_mse"] is not None
    assert entry["validation_rmse"] is not None
    assert entry["validation_mae"] is not None
    assert entry["prediction_std"] is not None
    assert "searched_body_depth" in entry
    assert "remaining_budget" in fb2

    fb3 = json.loads(third[third.index("{"): third.rindex("}") + 1])
    assert len(fb3["history"]) == 2  # FULL history, not just the last entry
    assert fb3["history"][1]["theta"] == [0.7]


# --- Test metrics never enter prompts ---------------------------------------


def test_protected_test_metrics_never_enter_prompts(tmp_path, train_val):
    store = _store(tmp_path)
    provider = _ScriptedProvider([_cand(0.3), _cand(0.7), _cand(-1.1)])
    run_closed_loop_arm_main(
        0, TASK, 3, 0, train_val, 3, store, provider, time.monotonic() + 60, feature_count=8,
    )
    store.close()
    for prompt in provider.prompts_seen:
        lowered = prompt.lower()
        assert "test_rmse" not in lowered
        assert "test rmse" not in lowered
        assert "protected" not in lowered


# --- Expressibility cache is keyed by architecture_hash ---------------------


def test_expressibility_cache_keyed_by_architecture_hash():
    from llm_vqc.free_amplitude.candidate_schema import architecture_ir
    from llm_vqc.free_amplitude.expressibility import IntrinsicDiagnosticsCache

    p1 = validate_complete_candidate(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0], "theta": 0.1}]}, 3
    ).proposal
    p2 = validate_complete_candidate(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0], "theta": 2.9}]}, 3
    ).proposal
    h1, h2 = architecture_hash(p1, 0), architecture_hash(p2, 0)
    assert h1 == h2  # same architecture despite different theta

    cache = IntrinsicDiagnosticsCache()
    d1 = cache.get_or_compute(architecture_ir(p1, 0), h1)
    d2 = cache.get_or_compute(architecture_ir(p2, 0), h2)
    assert len(cache) == 1  # ONE entry for both candidates
    assert d1 is d2
    assert d1.expressibility_kl == d2.expressibility_kl


# --- fixed-theta diagnostics are not labelled expressibility ----------------


def test_fixed_theta_diagnostics_not_labelled_expressibility():
    """`task_conditioned_entanglement` (the per-candidate fixed-theta
    diagnostic) must not carry any 'expressibility' label, and the
    expressibility module must expose no per-fixed-theta expressibility
    function."""
    import llm_vqc.free_amplitude.expressibility as ex

    # The task-conditioned result type has no expressibility field.
    fields = ex.EntanglementStats.__dataclass_fields__.keys()
    assert not any("express" in f.lower() for f in fields)

    # No public function computes 'expressibility' from a single fixed theta:
    # the only expressibility entry points require many sampled thetas.
    public = [n for n in dir(ex) if not n.startswith("_") and "express" in n.lower()]
    assert public == []  # expressibility appears only inside IntrinsicDiagnostics

    # And the intrinsic result records its many-sample provenance.
    assert ex.DIAGNOSTIC_STATE_SAMPLES > 1


# --- per-seed isolation ------------------------------------------------------


def test_per_seed_namespaces_and_results_are_independent(tmp_path):
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    store = _store(tmp_path)
    outcomes = []
    for seed in (0, 1):
        tv, _ = task.build(seed=seed)
        outcomes.append(
            run_random_arm_main(seed, TASK, 3, 0, tv, 2, store, time.monotonic() + 60)
        )
    store.close()
    assert outcomes[0].task_namespace != outcomes[1].task_namespace
    # Different seeds draw different data AND different random candidates.
    h0 = [c.candidate_hash for c in outcomes[0].candidates]
    h1 = [c.candidate_hash for c in outcomes[1].candidates]
    assert h0 != h1


# --- all required figure and source-data files are generated ----------------


REQUIRED_FIGURES = [
    "best_so_far_rmse", "validation_rmse_distribution", "test_rmse_by_arm",
    "prediction_vs_target", "complexity_vs_rmse", "closed_loop_theta_trajectory",
    "proposal_outcomes", "expressibility_by_arm", "entangling_capability_by_arm",
    "expressibility_vs_validation_rmse", "entanglement_vs_validation_rmse",
    "expressibility_vs_entanglement", "selected_fidelity_histograms",
    "selected_entanglement_distributions",
]


def test_all_required_figure_and_source_files_generated(tmp_path):
    from llm_vqc.free_amplitude import main_reporting
    from llm_vqc.free_amplitude.expressibility import IntrinsicDiagnosticsCache
    from llm_vqc.free_amplitude.main_runner import evaluate_protected_test_main
    from llm_vqc.free_amplitude.provenance import collect_provenance
    from llm_vqc.free_amplitude.provider import MockCompleteCandidateProvider
    from llm_vqc.free_amplitude.sampler import search_space_size_report

    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    store = _store(tmp_path)
    outcomes = []
    ds_diag = {}
    for seed in (0,):
        tv, d1 = task.build(seed=seed)
        test_split, d2 = task.build_test(seed=seed)
        ds_diag[seed] = {"train_val": d1, "test": d2}
        outcomes.append(run_random_arm_main(seed, TASK, 3, 0, tv, 2, store, time.monotonic() + 60))
        prov = MockCompleteCandidateProvider(seed, 3)
        outcomes.append(run_open_loop_arm_main(
            seed, TASK, 3, 0, tv, 2, store, prov, time.monotonic() + 60, feature_count=8))
        prov2 = MockCompleteCandidateProvider(seed, 3)
        outcomes.append(run_closed_loop_arm_main(
            seed, TASK, 3, 0, tv, 2, store, prov2, time.monotonic() + 60, feature_count=8))
        for o in outcomes:
            evaluate_protected_test_main(o, 0, test_split)
    store.close()

    out = tmp_path / "out"

    class _Args:
        max_gates = 5
        budget = 2
        seeds = 1
        dataset_profile = "amplitude_n3_smoke_v1"

    main_reporting.write_all_outputs(
        output_dir=out, args=_Args(), provenance=collect_provenance(pathlib.Path(".")),
        profile=AMPLITUDE_N3_SMOKE_V1, arm_outcomes=outcomes,
        dataset_diagnostics_by_seed=ds_diag,
        search_space_report=search_space_size_report(3, 5),
        diagnostics_cache=IntrinsicDiagnosticsCache(), task=task, readout_qubit=0,
        elapsed_seconds=1.0, run_label="MOCK TEST RUN",
    )

    for name in REQUIRED_FIGURES:
        for ext in ("png", "svg", "csv"):
            assert (out / "figures" / f"{name}.{ext}").exists(), f"missing figures/{name}.{ext}"
    # selected_circuits diagrams (at least the robust .txt for each arm/seed)
    diagrams = list((out / "figures" / "selected_circuits").glob("*.txt"))
    assert len(diagrams) >= 3
    for data_file in (
        "candidate_trace.csv", "selected_circuits.json", "architecture_diagnostics.csv",
        "cross_seed_summary.json", "EXPERIMENT_CONFIG.json", "report.md",
    ):
        assert (out / data_file).exists(), f"missing {data_file}"
