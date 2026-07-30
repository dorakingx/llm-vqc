"""Tests for the presentation verifier.

The verifier must (a) pass on the real package and (b) actually fail when
a deck drifts from its sources. Each negative test corrupts one thing in
a throwaway copy and asserts the matching failure is reported.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "docs" / "presentation" / "bench_v2_weekly_revised"

_spec = importlib.util.spec_from_file_location(
    "verify_bench_v2_presentation", REPO / "scripts" / "verify_bench_v2_presentation.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_bench_v2_presentation"] = _mod
_spec.loader.exec_module(_mod)


def _run(pkg: Path) -> tuple[int, list[str]]:
    problems: list[str] = []
    _mod.check_files(pkg, problems)
    pptx = pkg / "20260731_GSoC_revised.pptx"
    if pptx.is_file():
        texts = _mod.slide_texts(pptx)
        _mod.check_forbidden(texts, problems)
        _mod.check_status_consistency(texts, _mod.load_status(pkg), problems)
        _mod.check_numbers(pkg, texts, problems)
        _mod.check_metadata(pkg, texts, problems)
    _mod.check_claim_map(pkg, problems)
    return len(problems), problems


@pytest.fixture
def pkg_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "pkg"
    shutil.copytree(PKG, dst)
    return dst


def _rewrite_slide(pkg: Path, needle: str, replacement: str) -> None:
    """Rewrite text inside the packed pptx (round-trips the zip)."""
    src = pkg / "20260731_GSoC_revised.pptx"
    tmp = src.with_suffix(".tmp")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.startswith("ppt/slides/slide"):
                text = data.decode("utf-8")
                if needle in text:
                    text = text.replace(needle, replacement)
                data = text.encode("utf-8")
            zout.writestr(info.filename, data)
    tmp.replace(src)


def test_real_package_passes():
    count, problems = _run(PKG)
    assert count == 0, problems


def test_slide_texts_finds_all_slides():
    texts = _mod.slide_texts(PKG / "20260731_GSoC_revised.pptx")
    assert len(texts) >= 10
    assert any("Interim update" in t for t in texts)


def test_fails_when_a_required_artifact_is_removed(pkg_copy: Path):
    (pkg_copy / "speaker_notes.md").unlink()
    _, problems = _run(pkg_copy)
    assert any("speaker_notes.md" in p for p in problems)


def test_fails_when_a_required_figure_is_removed(pkg_copy: Path):
    (pkg_copy / "figures" / "fig_e3_scaling.png").unlink()
    _, problems = _run(pkg_copy)
    assert any("fig_e3_scaling.png" in p for p in problems)


def test_fails_on_preregistered(pkg_copy: Path):
    _rewrite_slide(pkg_copy, "protocol-frozen", "preregistered")
    _, problems = _run(pkg_copy)
    assert any("preregistered" in p for p in problems)


def test_fails_on_equivalence_language(pkg_copy: Path):
    _rewrite_slide(pkg_copy, "No Holm-adjusted difference was detected",
                   "The arms are statistically indistinguishable")
    _, problems = _run(pkg_copy)
    assert any("indistinguishable" in p for p in problems)


def test_fails_on_unqualified_hardware_cost(pkg_copy: Path):
    # The rule is "hardware cost" without the word "proxy" on the same
    # slide, so the qualifier has to be removed as well as the phrase
    # introduced — replacing it in place does exactly that.
    _rewrite_slide(pkg_copy, "A hardware-relevant proxy", "The hardware cost")
    _, problems = _run(pkg_copy)
    assert any("hardware cost" in p for p in problems)


def test_qualified_hardware_cost_is_accepted(pkg_copy: Path):
    """The same phrase is fine when the slide also calls it a proxy."""
    _rewrite_slide(pkg_copy, "Median compiled 2-qubit gates",
                   "Median hardware cost in gates")
    _, problems = _run(pkg_copy)
    assert not any("hardware cost" in p for p in problems)


def test_fails_on_stale_cell_count(pkg_copy: Path):
    src = pkg_copy / "data" / "fig_matrix.csv"
    src.write_text(src.read_text().replace("280,280", "280,279"))
    _, problems = _run(pkg_copy)
    assert any("classical count" in p for p in problems)


def test_fails_on_invented_p_value(pkg_copy: Path):
    _rewrite_slide(pkg_copy, "smallest p = 0.071", "smallest p = 0.001")
    _, problems = _run(pkg_copy)
    assert any("0.001" in p or "0.071" in p for p in problems)


def test_fails_when_benchmark_called_complete(pkg_copy: Path):
    _rewrite_slide(pkg_copy, "the benchmark as a whole is not complete",
                   "the benchmark is complete")
    _, problems = _run(pkg_copy)
    assert any("calls the benchmark complete" in p for p in problems)


def test_fails_on_placeholder_text(pkg_copy: Path):
    _rewrite_slide(pkg_copy, "Interim update", "TODO update")
    _, problems = _run(pkg_copy)
    assert any("placeholder" in p for p in problems)


def test_fails_on_commit_metadata_mismatch(pkg_copy: Path):
    manifest = pkg_copy / "artifact_manifest.json"
    doc = json.loads(manifest.read_text())
    doc["source_commit"] = "0" * 40
    manifest.write_text(json.dumps(doc, indent=2))
    _, problems = _run(pkg_copy)
    assert any("source commit" in p for p in problems)


def test_fails_on_claim_with_missing_source(pkg_copy: Path):
    path = pkg_copy / "claim_source_map.yaml"
    path.write_text(path.read_text().replace(
        "source: llm_vqc/free_amplitude/model.py",
        "source: llm_vqc/does_not_exist.py", 1))
    _, problems = _run(pkg_copy)
    assert any("claim source not found" in p for p in problems)
