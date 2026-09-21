#!/usr/bin/env python
"""Prove that every completed classical candidate obeys the strict search
grammar that the LLM arms will be held to.

The strict rules (protocol §4 as tightened before the real-LLM run):

  R1  a rotation operation carries exactly one gate;
  R2  an entangling operation uses `wires: "all"` — the only policy the
      frozen classical sampler can emit;
  R3  a `pairs` entangler carries 1..floor(n/2) disjoint, non-duplicate
      two-element pairs;
  R4  no operation packs several gate layers into one op.

R1 and R4 are the same rule seen from two sides: a multi-gate rotation
layer *is* a compact packing of several physical layers into one
operation, and it would let an LLM buy more circuit per unit of the
`max_ops = min(2n, 16)` budget than any classical arm can.

Exit code 0 means every search-arm candidate already complies, so
enforcing the rule on new proposals cannot change the completed
comparison. Non-zero means the run must stop and the discrepancy be
documented instead of silently re-defining the comparison.

Fixed reference templates are reported separately: they are anchors, not
search candidates, and are deliberately allowed outside the grammar.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sqlite3
import sys
from pathlib import Path

SEARCH_ARMS = {"random_structure", "evolutionary_structure", "greedy_growth"}


def classify(arm: str) -> str:
    if arm in SEARCH_ARMS:
        return "search"
    if arm.startswith("ref_"):
        return "reference"
    if "llm" in arm:
        return "llm"
    return "other"


def check_operation(op: dict, n_qubits: int | None) -> list[str]:
    """Return the rule ids this operation violates."""
    bad: list[str] = []
    if op.get("type") == "rot":
        if len(op.get("gates", [])) != 1:
            bad.append("R1_rot_multi_gate")
    elif op.get("type") == "entangle":
        if op.get("wires") != "all":
            bad.append("R2_entangle_wires_not_all")
        if op.get("pattern") == "pairs":
            pairs = op.get("pairs") or []
            flat = [w for p in pairs for w in p]
            limit = (n_qubits or 0) // 2
            ok = (
                1 <= len(pairs) <= limit
                and all(len(p) == 2 for p in pairs)
                and len(set(flat)) == len(flat)
            )
            if not ok:
                bad.append("R3_pairs_invalid")
    return bad


def scan(runs_root: Path) -> dict:
    counts: collections.Counter = collections.Counter()
    violations: collections.Counter = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    for manifest_path in sorted(glob.glob(str(runs_root / "E[123]" / "cells" / "*.json"))):
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("status") != "complete":
            continue
        kind = classify(manifest["arm"])
        store = Path(manifest["store_path"])
        if not store.is_file():
            continue
        counts[f"{kind}:cells"] += 1
        con = sqlite3.connect(store)
        rows = con.execute(
            "SELECT trained_weights_json FROM evaluations "
            "WHERE trained_weights_json IS NOT NULL"
        ).fetchall()
        con.close()
        for (blob,) in rows:
            payload = json.loads(blob)
            ops = payload.get("operations")
            if not ops:
                continue
            counts[f"{kind}:candidates"] += 1
            n_qubits = payload.get("n_qubits")
            for op in ops:
                counts[f"{kind}:operations"] += 1
                for rule in check_operation(op, n_qubits):
                    violations[f"{kind}:{rule}"] += 1
                    if len(examples[f"{kind}:{rule}"]) < 3:
                        examples[f"{kind}:{rule}"].append(
                            {"arm": manifest["arm"], "n_qubits": n_qubits, "op": op}
                        )
    return {
        "counts": dict(sorted(counts.items())),
        "violations": dict(sorted(violations.items())),
        "examples": {k: v for k, v in examples.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "runs" / "bench_v2")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = scan(args.runs_root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    print("Search-space parity scan")
    for key, value in report["counts"].items():
        print(f"  {key:28} {value}")
    search_violations = {k: v for k, v in report["violations"].items()
                         if k.startswith("search:") or k.startswith("llm:")}
    reference_violations = {k: v for k, v in report["violations"].items()
                            if k.startswith("reference:")}
    print()
    if reference_violations:
        print("Fixed-reference templates outside the search grammar (expected, anchors):")
        for key, value in reference_violations.items():
            print(f"  {key:34} {value}")
        print()
    if search_violations:
        print("STOP — searched candidates violate the strict grammar:")
        for key, value in search_violations.items():
            print(f"  FAIL {key:32} {value}")
            for ex in report["examples"].get(key, []):
                print(f"       {ex}")
        return 1
    print("PASS — every searched candidate already satisfies R1-R4; "
          "enforcing them on new proposals cannot change the completed comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
