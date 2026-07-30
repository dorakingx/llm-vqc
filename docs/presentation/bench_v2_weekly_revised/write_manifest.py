#!/usr/bin/env python
"""Regenerate artifact_manifest.json for this deck package.

The manifest was previously maintained by hand, which let its
`source_commit` drift away from the commit `generate_deck.js` stamps on
the title slide. Both now read the same `git rev-parse HEAD`, so the
verifier's cross-check cannot pass by accident and cannot fail because
someone forgot to edit a JSON field.

Run it after the deck, PDF and renders exist:

    python docs/presentation/bench_v2_weekly_revised/write_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "artifact_manifest.json"
#: hashed as content, not as build scratch
SKIP_DIRS = {"__pycache__"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    head = git("rev-parse", "HEAD")

    files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if SKIP_DIRS & set(path.relative_to(HERE).parts):
            continue
        files.append({
            "path": str(path.relative_to(HERE)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })

    doc = {
        "package": "docs/presentation/bench_v2_weekly_revised",
        "deck_stem": previous.get("deck_stem", "20260731_GSoC_revised"),
        "presentation_date": previous.get("presentation_date", "July 31, 2026"),
        "source_commit": head,
        "base_commit": head,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "generation": [
            "python docs/presentation/bench_v2_weekly_revised/build_figures.py",
            "node docs/presentation/bench_v2_weekly_revised/generate_deck.js",
            "python docs/presentation/bench_v2_weekly_revised/recompress_pptx.py "
            "20260731_GSoC_revised.pptx",
            "soffice --headless --convert-to pdf 20260731_GSoC_revised.pptx",
            "pdftoppm -png -r 130 20260731_GSoC_revised.pdf renders/slide",
            "python docs/presentation/bench_v2_weekly_revised/write_manifest.py",
        ],
        "note": "artifact_manifest.json itself is excluded from the file list "
                "it describes; source_commit is read from git at build time and "
                "is the same value generate_deck.js stamps on the title slide",
        "files": files,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"artifact_manifest.json: {len(files)} files at {head[:7]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
