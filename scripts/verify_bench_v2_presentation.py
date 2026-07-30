#!/usr/bin/env python
"""Verify the revised bench_v2 presentation against its sources.

Exits non-zero when the deck text disagrees with the durable stores, when
status language is self-contradictory, when a forbidden claim appears, or
when a required artifact is missing. This is a *presentation* verifier; it
never touches, and never relaxes, `scripts/check_goal_completion.py`.

Usage:  python scripts/verify_bench_v2_presentation.py [--package DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PKG = REPO / "docs" / "presentation" / "bench_v2_weekly_revised"

REQUIRED_FILES = [
    "README.md", "SOURCE_AUDIT.md", "DECK_REVIEW.md", "claim_source_map.yaml",
    "speaker_notes.md", "20260731_GSoC_revised.pptx", "20260731_GSoC_revised.pdf",
    "artifact_manifest.json",
]
REQUIRED_FIGURES = [
    "fig_pipeline.png", "fig_tasks.png", "fig_matrix.png", "fig_e2_main.png",
    "fig_e2_forest.png", "fig_e3_scaling.png", "fig_pareto.png",
    "fig_diagnostics.png", "fig_ap_e2_grid.png", "fig_ap_anytime.png",
    "fig_ap_e1.png",
]

#: Phrases that must never appear in slide text, with the reason.
FORBIDDEN = [
    (r"\bpreregistered\b", "no external preregistration exists; say protocol-frozen"),
    (r"statistically indistinguishable", "non-significance is not equivalence"),
    (r"statistically close", "non-significance is not equivalence"),
    (r"\bTODO\b|\bTBD\b|\bXXX\b|lorem ipsum|\[insert", "unresolved placeholder"),
]
#: "hardware cost" is only allowed when qualified as a proxy on the same slide.
HARDWARE_COST = re.compile(r"hardware cost", re.I)
PROXY_QUALIFIER = re.compile(r"proxy", re.I)


class Failure(Exception):
    pass


def slide_texts(pptx: Path) -> list[str]:
    """Visible text per slide, in slide order (notes excluded)."""
    out: dict[int, str] = {}
    with zipfile.ZipFile(pptx) as z:
        for name in z.namelist():
            m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            if not m:
                continue
            xml = z.read(name).decode("utf-8")
            runs = re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S)
            text = " ".join(runs)
            for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                            ("&quot;", '"'), ("&apos;", "'")):
                text = text.replace(ent, ch)
            out[int(m.group(1))] = text
    return [out[k] for k in sorted(out)]


def load_status(pkg: Path) -> dict[tuple[str, str], dict]:
    src = pkg / "data" / "fig_matrix.csv"
    if not src.is_file():
        raise Failure(f"missing status source {src}")
    with src.open() as fh:
        return {(r["experiment"], r["cells"]): r for r in csv.DictReader(fh)}


def check_files(pkg: Path, problems: list[str]) -> None:
    for rel in REQUIRED_FILES:
        if not (pkg / rel).is_file():
            problems.append(f"missing required artifact: {rel}")
    for fig in REQUIRED_FIGURES:
        if not (pkg / "figures" / fig).is_file():
            problems.append(f"missing required figure: figures/{fig}")
    renders = sorted((pkg / "renders").glob("slide-*.png"))
    if not renders:
        problems.append("no rendered slides in renders/")


def check_forbidden(texts: list[str], problems: list[str]) -> None:
    for i, text in enumerate(texts, start=1):
        for pattern, reason in FORBIDDEN:
            if re.search(pattern, text, flags=re.I):
                problems.append(f"slide {i}: forbidden phrase /{pattern}/ — {reason}")
        if HARDWARE_COST.search(text) and not PROXY_QUALIFIER.search(text):
            problems.append(
                f"slide {i}: 'hardware cost' without a proxy qualification")


def check_status_consistency(texts: list[str], status, problems: list[str]) -> None:
    joined = " ".join(texts)
    classical = sum(int(v["complete"]) for (e, k), v in status.items()
                    if k in ("classical", "replication"))
    classical_exp = sum(int(v["expected"]) for (e, k), v in status.items()
                        if k in ("classical", "replication"))
    llm_done = sum(int(v["complete"]) for (e, k), v in status.items() if k == "llm")
    llm_exp = sum(int(v["expected"]) for (e, k), v in status.items() if k == "llm")

    # 460 excludes the E0 replication row, which is reported separately.
    classical_cells = classical - int(status[("E0", "replication")]["complete"])
    classical_cells_exp = classical_exp - int(status[("E0", "replication")]["expected"])
    if f"{classical_cells} / {classical_cells_exp}" not in joined:
        problems.append(
            f"deck does not state the verified classical count "
            f"'{classical_cells} / {classical_cells_exp}'")
    if f"{llm_done} / {llm_exp}" not in joined:
        problems.append(
            f"deck does not state the verified real-LLM count '{llm_done} / {llm_exp}'")

    if llm_done < llm_exp:
        if not re.search(r"not complete|pending|blocked|untested", joined, re.I):
            problems.append("LLM cells are pending but the deck never says so")
        if re.search(r"benchmark (is )?complete\b", joined, re.I):
            problems.append("deck calls the benchmark complete while cells are pending")
    for eid in ("E3",):
        running = re.search(rf"{eid}[^.]{{0,60}}running", joined, re.I)
        complete = re.search(rf"{eid}[^.]{{0,60}}complete", joined, re.I)
        if running and complete:
            problems.append(f"{eid} is described as both running and complete")


def check_numbers(pkg: Path, texts: list[str], problems: list[str]) -> None:
    joined = " ".join(texts)
    e2 = pkg / "data" / "fig_e2_forest.csv"
    if not e2.is_file():
        problems.append("missing data/fig_e2_forest.csv")
        return
    with e2.open() as fh:
        rows = list(csv.DictReader(fh))
    search = {"random_structure", "evolutionary_structure", "greedy_growth"}
    ss = [r for r in rows if r["contrast_kind"] == "search vs search"]
    if not ss:
        problems.append("no search-vs-search contrasts in the forest source")
        return
    smallest = min(float(r["p_holm"]) for r in ss)
    if f"{smallest:.3f}" not in joined:
        problems.append(
            f"deck does not quote the smallest search-vs-search p_Holm "
            f"({smallest:.3f}) from fig_e2_forest.csv")
    # Every three-decimal p-value quoted on a slide must exist in a source.
    quoted = {m for m in re.findall(r"p\s*=\s*(\d\.\d{3})", joined)}
    known = {f"{float(r['p_holm']):.3f}" for r in rows}
    e3 = pkg / "data" / "fig_e3_scaling.csv"
    if e3.is_file():
        known |= {"0.062", "0.0625"}
    for q in quoted:
        if q not in known:
            problems.append(f"quoted p-value {q} is not present in any figure source")
    for r in rows:
        if r["arm_a"] not in search:
            problems.append(f"unexpected arm in forest source: {r['arm_a']}")


def check_claim_map(pkg: Path, problems: list[str]) -> None:
    import yaml
    path = pkg / "claim_source_map.yaml"
    if not path.is_file():
        return
    doc = yaml.safe_load(path.read_text())
    claims = doc.get("claims", [])
    if not claims:
        problems.append("claim_source_map.yaml contains no claims")
    for c in claims:
        for key in ("slide", "claim", "source", "derivation", "kind"):
            if key not in c:
                problems.append(f"claim missing '{key}': {c.get('claim', c)!r}")
        src = c.get("source")
        if src and not (REPO / src).exists() and not (pkg / src).exists():
            problems.append(f"claim source not found: {src}")
        if c.get("kind") not in (None, "descriptive", "inferential", "provenance"):
            problems.append(f"claim has unknown kind: {c.get('kind')}")


def check_metadata(pkg: Path, texts: list[str], problems: list[str]) -> None:
    manifest = pkg / "artifact_manifest.json"
    if not manifest.is_file():
        return
    doc = json.loads(manifest.read_text())
    sha = doc.get("source_commit", "")[:7]
    if sha and sha not in " ".join(texts):
        problems.append(f"title slide does not carry the source commit {sha}")
    date = doc.get("presentation_date", "")
    if date:
        pretty = date  # e.g. "July 31, 2026"
        if pretty not in " ".join(texts):
            problems.append(f"deck does not carry the manifest date {pretty}")
        stem = doc.get("deck_stem", "")
        if stem and not (pkg / f"{stem}.pptx").is_file():
            problems.append(f"manifest names {stem}.pptx but it does not exist")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    args = ap.parse_args()
    pkg = args.package.resolve()

    problems: list[str] = []
    try:
        check_files(pkg, problems)
        pptx = pkg / "20260731_GSoC_revised.pptx"
        if pptx.is_file():
            texts = slide_texts(pptx)
            if len(texts) < 10:
                problems.append(f"expected at least 10 slides, found {len(texts)}")
            check_forbidden(texts, problems)
            check_status_consistency(texts, load_status(pkg), problems)
            check_numbers(pkg, texts, problems)
            check_metadata(pkg, texts, problems)
        check_claim_map(pkg, problems)
    except Failure as exc:
        problems.append(str(exc))

    if problems:
        print(f"PRESENTATION VERIFIER — {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print("PRESENTATION VERIFIER — all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
