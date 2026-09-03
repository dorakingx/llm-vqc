"""Guards on the redesigned deck build: the glossary and structure audits
must be able to fail, and the six-category grid contract must hold.
"""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("scripts/qae/build_qae_robustness_slides.py")
GRID = Path("scripts/qae/build_qae_robustness_grid_figures.py")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


slides = _load(SCRIPT, "qae_slides")
grid = _load(GRID, "qae_grid")


def test_glossary_audit_flags_a_use_before_its_definition():
    texts = ["We measured F_trash on slide one.",
             "Metric — trash fidelity:  F_trash = ⟨00| ρ_trash |00⟩"]
    problems = slides.glossary_audit(texts)
    assert any("'trash fidelity'" in p and "before its definition" in p for p in problems)


def test_glossary_audit_flags_a_term_that_is_never_defined():
    problems = slides.glossary_audit(["closed-loop redesigns everywhere", ""])
    assert any("'closed-loop'" in p and "never defined" in p for p in problems)


def test_glossary_audit_accepts_definition_on_the_same_slide():
    text = ("budget B = 8 ... each may train and score exactly B candidate circuits "
            "per seed (B = the evaluation budget)")
    assert not [p for p in slides.glossary_audit([text]) if "'budget B'" in p]


def test_six_categories_in_the_required_order():
    labels = [label.replace("\n", " ") for _m, label, *_ in grid.CATEGORIES]
    assert labels == ["Random", "Greedy", "LLM-Open Model A", "LLM-Closed Model A",
                      "LLM-Open Model B", "LLM-Closed Model B"]
    models = [model for _m, _l, model, *_ in grid.CATEGORIES]
    assert models == [None, None, "A", "A", "B", "B"]
    filled = [f for *_rest, f in grid.CATEGORIES]
    assert filled == [True, True, True, True, False, False]


def test_grid_has_nine_cells_per_hamiltonian_and_never_invents_data():
    coverage = json.loads(Path("outputs/qae_robustness/grid_cells.json").read_text())
    for family in ("TFIM", "XXZ"):
        assert len(coverage["coverage"][family]) == 9
    # the one-factor design: Model B exists only in the baseline cell
    for family, cells in coverage["coverage"].items():
        for cell, present in cells.items():
            if 4 in present or 5 in present:
                assert (family, cell) == ("TFIM", "4q_B8")


def test_the_deck_has_exactly_ten_slide_builders_in_the_required_order():
    source = SCRIPT.read_text()
    import re
    builders = re.findall(r"^def (slide_\d\d_\w+)\(", source, flags=re.M)
    assert len(builders) == 10
    assert builders == sorted(builders)
    ordered = source[source.index("for builder in ("):source.index("):\n            builder")]
    for name in builders:
        assert name in ordered


@pytest.mark.parametrize("text", [
    "target accuracy and minimum required budget",
])
def test_next_action_phrases_are_present_in_slide_nine_source(text):
    source = SCRIPT.read_text()
    start = source.index("def slide_09_limits_next")
    end = source.index("def slide_10_conclusion")
    assert text in source[start:end]
