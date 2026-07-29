#!/usr/bin/env python
"""Machine-checkable completion gate for the bench_v2 goal.

Reads `GOAL_CONTRACT.yaml` + `configs/bench_v2/protocol_v2.yaml`, verifies
every mandatory criterion against durable stores (`runs/bench_v2/`),
committed artifacts (`outputs/bench_v2/`), and the source tree, and exits
0 only if ALL criteria pass. Any missing, interrupted, or silently
excluded cell is a hard failure with an explicit reason.

Usage:
    python scripts/check_goal_completion.py [--repo-root PATH] [--skip-commands]

`--skip-commands` skips the (slow) full pytest/ruff criterion C15 for
interim progress checks; the final completion claim must run WITHOUT it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CheckResult:
    criterion_id: str
    passed: bool
    details: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContractChecker:
    def __init__(self, repo_root: Path, skip_commands: bool = False) -> None:
        self.root = repo_root
        self.skip_commands = skip_commands
        self.contract = yaml.safe_load((repo_root / "GOAL_CONTRACT.yaml").read_text())
        self.protocol = yaml.safe_load(
            (repo_root / self.contract["protocol"]).read_text()
        )

    # ------------------------------------------------------------------
    # Expected-cell enumeration from the frozen protocol matrix
    # ------------------------------------------------------------------

    def expected_cells(self, eid: str) -> list[dict]:
        exp = self.protocol["experiments"][eid]
        if "tasks" not in exp or "arms" not in exp:
            return []
        cells = []
        replicates = int(exp.get("replicates", 0))
        for task in exp["tasks"]:
            for n in exp.get("n_qubits", [None]):
                for arm in exp["arms"]:
                    for r in range(replicates):
                        cells.append(
                            {
                                "experiment": eid,
                                "task": task,
                                "n_qubits": n,
                                "arm": arm,
                                "replicate": r,
                                "cell_id": f"{task}_{n}q_{arm}_r{r}",
                            }
                        )
        return cells

    def load_cell_manifest(self, eid: str, cell_id: str) -> dict | None:
        path = self.root / "runs" / "bench_v2" / eid / "cells" / f"{cell_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Individual criterion checks
    # ------------------------------------------------------------------

    def check_file_exists_nonempty(self, crit: dict) -> CheckResult:
        path = self.root / crit["path"]
        ok = path.is_file() and path.stat().st_size > 0
        return CheckResult(crit["id"], ok, [] if ok else [f"missing/empty: {crit['path']}"])

    def check_files_exist_and_hashes_recorded(self, crit: dict) -> CheckResult:
        details: list[str] = []
        for rel in crit["paths"]:
            if not (self.root / rel).is_file():
                details.append(f"missing: {rel}")
        record = self.root / crit["hash_record"]
        if not record.is_file():
            details.append(f"missing hash record: {crit['hash_record']}")
        else:
            recorded = json.loads(record.read_text())
            for rel in crit["paths"]:
                path = self.root / rel
                if path.is_file():
                    actual = sha256_file(path)
                    if recorded.get(rel) != actual:
                        details.append(
                            f"hash mismatch for {rel}: recorded "
                            f"{recorded.get(rel)!r} != actual {actual[:16]}..."
                        )
        return CheckResult(crit["id"], not details, details)

    def check_signal_suite_complete(self, crit: dict) -> CheckResult:
        details: list[str] = []
        pkg = self.root / crit["scope"]
        if not pkg.is_dir():
            return CheckResult(crit["id"], False, [f"missing package: {crit['scope']}"])
        source = "\n".join(
            p.read_text() for p in sorted(pkg.rglob("*.py"))
        )
        for task in crit["tasks"]:
            if task not in source:
                details.append(f"task {task!r} not found in {crit['scope']}")
        for forbidden in crit["forbidden_imports"]:
            if forbidden.lower() in source.lower():
                details.append(f"forbidden reference {forbidden!r} present in signal_suite")
        return CheckResult(crit["id"], not details, details)

    def check_modules_exist(self, crit: dict) -> CheckResult:
        details = []
        for module in crit["modules"]:
            rel = Path(*module.split(".")).with_suffix(".py")
            pkg_init = Path(*module.split(".")) / "__init__.py"
            if not (self.root / rel).is_file() and not (self.root / pkg_init).is_file():
                details.append(f"module not found: {module}")
        return CheckResult(crit["id"], not details, details)

    def check_arms_registered(self, crit: dict) -> CheckResult:
        registry = self.root / "llm_vqc" / "bench_v2" / "arm_registry.py"
        if not registry.is_file():
            return CheckResult(crit["id"], False, ["missing llm_vqc/bench_v2/arm_registry.py"])
        text = registry.read_text()
        details = [
            f"arm not registered: {arm}"
            for arm in [*crit["track_a"], *crit["track_b"]]
            if f'"{arm}"' not in text
        ]
        return CheckResult(crit["id"], not details, details)

    def check_store_cells_exist(self, crit: dict) -> CheckResult:
        eid = crit["experiment"]
        need_n = crit["require_n_qubits"]
        min_rep = crit["min_replicates"]
        details = []
        found = 0
        for cell in self.expected_cells(eid):
            if cell["n_qubits"] != need_n:
                continue
            manifest = self.load_cell_manifest(eid, cell["cell_id"])
            if manifest is not None and manifest.get("status") == "complete":
                found += 1
            elif cell["replicate"] < min_rep:
                details.append(f"{eid} missing/incomplete n={need_n} cell: {cell['cell_id']}")
        if found == 0:
            details.append(f"no complete n={need_n} cells found for {eid}")
        return CheckResult(crit["id"], not details, details)

    def check_matrix_complete(self, crit: dict) -> CheckResult:
        details = []
        for eid in crit["experiments"]:
            exp = self.protocol["experiments"][eid]
            budget_key = "budget_unique" if "budget_unique" in exp else "budget"
            for cell in self.expected_cells(eid):
                manifest = self.load_cell_manifest(eid, cell["cell_id"])
                if manifest is None:
                    details.append(f"{eid}: missing cell manifest {cell['cell_id']}")
                    continue
                if manifest.get("status") != "complete":
                    details.append(
                        f"{eid}: cell {cell['cell_id']} status="
                        f"{manifest.get('status')!r} (interrupted cells are not complete)"
                    )
                    continue
                got_budget = manifest.get("budget_unique")
                if got_budget != exp.get(budget_key):
                    details.append(
                        f"{eid}: cell {cell['cell_id']} budget {got_budget} "
                        f"!= protocol {exp.get(budget_key)}"
                    )
                r = cell["replicate"]
                stats = self.protocol["statistics"]
                if manifest.get("data_seed") != stats["data_seed_base"] + r:
                    details.append(f"{eid}: {cell['cell_id']} wrong data_seed (pairing broken)")
                if manifest.get("search_seed") != stats["search_seed_base"] + r:
                    details.append(f"{eid}: {cell['cell_id']} wrong search_seed (pairing broken)")
        return CheckResult(crit["id"], not details, details[:60])

    def check_llm_cells_real_provider(self, crit: dict) -> CheckResult:
        details = []
        for eid in crit["experiments"]:
            for cell in self.expected_cells(eid):
                if "llm" not in cell["arm"]:
                    continue
                manifest = self.load_cell_manifest(eid, cell["cell_id"])
                if manifest is None:
                    details.append(f"{eid}: LLM cell missing: {cell['cell_id']}")
                    continue
                llm = manifest.get("llm") or {}
                if llm.get("mock", True):
                    details.append(f"{eid}: {cell['cell_id']} used a MOCK provider")
                if not llm.get("model_snapshot"):
                    details.append(f"{eid}: {cell['cell_id']} lacks model snapshot")
                if int(llm.get("successful_calls", 0)) < 1:
                    details.append(f"{eid}: {cell['cell_id']} has zero successful LLM calls")
        return CheckResult(crit["id"], not details, details[:60])

    def check_test_gate_discipline(self, crit: dict) -> CheckResult:
        details = []
        for eid in crit["experiments"]:
            for cell in self.expected_cells(eid):
                manifest = self.load_cell_manifest(eid, cell["cell_id"])
                if manifest is None:
                    details.append(f"{eid}: missing manifest {cell['cell_id']}")
                    continue
                gate = manifest.get("test_gate") or {}
                if gate.get("evaluation_count") != 1:
                    details.append(
                        f"{eid}: {cell['cell_id']} test gate count "
                        f"{gate.get('evaluation_count')!r} != 1"
                    )
                sel_ts = manifest.get("selection_timestamp")
                gate_ts = gate.get("timestamp")
                if not sel_ts or not gate_ts or not (gate_ts >= sel_ts):
                    details.append(
                        f"{eid}: {cell['cell_id']} test-gate timestamp not after selection"
                    )
        return CheckResult(crit["id"], not details, details[:60])

    def check_artifacts_exist(self, crit: dict) -> CheckResult:
        details = []
        for pattern in crit["patterns"]:
            base = self.root
            matches = list(base.glob(pattern))
            if not matches:
                details.append(f"no artifact matches pattern: {pattern}")
        return CheckResult(crit["id"], not details, details)

    def check_resource_metrics_present(self, crit: dict) -> CheckResult:
        path = self.root / "outputs" / "bench_v2" / "E2" / "resource_metrics.csv"
        if not path.is_file():
            return CheckResult(
                crit["id"], False, ["missing outputs/bench_v2/E2/resource_metrics.csv"]
            )
        header = path.read_text().splitlines()[0] if path.stat().st_size else ""
        details = [f"column missing: {col}" for col in crit["require"] if col not in header]
        return CheckResult(crit["id"], not details, details)

    def check_figures_have_source_data(self, crit: dict) -> CheckResult:
        root = self.root / crit["outputs_root"]
        if not root.is_dir():
            return CheckResult(crit["id"], False, [f"missing {crit['outputs_root']}"])
        details = []
        for fig in list(root.rglob("*.png")) + list(root.rglob("*.svg")):
            stem = fig.with_suffix("")
            has_source = any(
                stem.with_suffix(ext).is_file() for ext in (".csv", ".json")
            ) or (fig.parent / "figure_sources.json").is_file()
            if not has_source:
                details.append(f"figure without source data: {fig.relative_to(self.root)}")
        return CheckResult(crit["id"], not details, details[:40])

    def check_commands_succeed(self, crit: dict) -> CheckResult:
        if self.skip_commands:
            return CheckResult(
                crit["id"], False,
                ["SKIPPED via --skip-commands (final completion must run this)"],
            )
        venv_python = self.root.parent.parent.parent / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = self.root / ".venv" / "bin" / "python"
        details = []
        for template in crit["commands"]:
            cmd = template.replace(".venv-relative", str(venv_python) + " -m")
            proc = subprocess.run(
                cmd, shell=True, cwd=self.root, capture_output=True, text=True,
                timeout=3600,
            )
            if proc.returncode != 0:
                tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
                details.append(f"command failed ({cmd}): " + " | ".join(tail))
        return CheckResult(crit["id"], not details, details)

    def check_report_contains(self, crit: dict) -> CheckResult:
        path = self.root / crit["path"]
        if not path.is_file():
            return CheckResult(crit["id"], False, [f"missing report: {crit['path']}"])
        text = path.read_text()
        details = [
            f"report lacks required phrase: {phrase!r}"
            for phrase in crit["must_contain"]
            if phrase.lower() not in text.lower()
        ]
        advantage_pattern = re.compile(
            r"(demonstrates|shows|proves|establishes)\s+quantum\s+advantage", re.I
        )
        if advantage_pattern.search(text):
            details.append("report appears to claim quantum advantage (prohibited)")
        return CheckResult(crit["id"], not details, details)

    def check_no_incomplete_cells(self, crit: dict) -> CheckResult:
        details = []
        runs_root = self.root / "runs" / "bench_v2"
        for eid in crit["experiments"]:
            exp_dir = runs_root / eid
            cells_dir = exp_dir / "cells"
            if eid in ("E0", "E4", "E5"):
                marker = self.root / "outputs" / "bench_v2" / eid / "COMPLETE.json"
                if not marker.is_file():
                    rel = marker.relative_to(self.root)
                    details.append(f"{eid}: completion marker missing: {rel}")
                continue
            if not cells_dir.is_dir():
                details.append(f"{eid}: no cell manifests at all")
                continue
            for manifest_path in sorted(cells_dir.glob("*.json")):
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "complete":
                    details.append(
                        f"{eid}: {manifest_path.stem} status={manifest.get('status')!r}"
                    )
        return CheckResult(crit["id"], not details, details[:60])

    # ------------------------------------------------------------------

    CHECKS = {
        "file_exists_nonempty": check_file_exists_nonempty,
        "files_exist_and_hashes_recorded": check_files_exist_and_hashes_recorded,
        "signal_suite_complete": check_signal_suite_complete,
        "modules_exist": check_modules_exist,
        "arms_registered": check_arms_registered,
        "store_cells_exist": check_store_cells_exist,
        "matrix_complete": check_matrix_complete,
        "llm_cells_real_provider": check_llm_cells_real_provider,
        "test_gate_discipline": check_test_gate_discipline,
        "artifacts_exist": check_artifacts_exist,
        "resource_metrics_present": check_resource_metrics_present,
        "figures_have_source_data": check_figures_have_source_data,
        "commands_succeed": check_commands_succeed,
        "report_contains": check_report_contains,
        "no_incomplete_cells": check_no_incomplete_cells,
    }

    def run(self) -> int:
        results: list[CheckResult] = []
        for crit in self.contract["criteria"]:
            fn = self.CHECKS.get(crit["check"])
            if fn is None:
                results.append(
                    CheckResult(crit["id"], False, [f"unknown check kind {crit['check']!r}"])
                )
                continue
            try:
                results.append(fn(self, crit))
            except Exception as exc:  # a crashing check is a failing check
                results.append(
                    CheckResult(crit["id"], False, [f"check crashed: {type(exc).__name__}: {exc}"])
                )

        passed = sum(1 for r in results if r.passed)
        print(f"GOAL CONTRACT CHECK — {passed}/{len(results)} criteria pass\n")
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.criterion_id}")
            for line in r.details:
                print(f"       - {line}")
        if passed == len(results):
            print("\nALL CRITERIA PASS — goal completion is verifiable.")
            return 0
        print(f"\n{len(results) - passed} criteria FAIL — the goal is NOT complete.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--skip-commands", action="store_true")
    args = parser.parse_args()
    checker = ContractChecker(args.repo_root.resolve(), skip_commands=args.skip_commands)
    return checker.run()


if __name__ == "__main__":
    sys.exit(main())
