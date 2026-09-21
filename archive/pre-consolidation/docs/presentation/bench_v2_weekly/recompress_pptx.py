#!/usr/bin/env python
"""Recompress the generated .pptx with deflate.

pptxgenjs writes every zip entry with STORE (no compression), so the
slide XML — ~150 KB of it — ships uncompressed. Rewriting the same
entries, in the same order, with ZIP_DEFLATED is a lossless container
change (PowerPoint, LibreOffice and Google Slides all read deflate) that
roughly halves the file. Size matters here because the deck is uploaded
to Google Drive as a single base64 payload.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path


def recompress(path: Path) -> tuple[int, int]:
    before = path.stat().st_size
    tmp = path.with_suffix(".pptx.tmp")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as dst:
        for info in src.infolist():  # preserve original entry order
            dst.writestr(info.filename, src.read(info.filename))
    shutil.move(tmp, path)
    return before, path.stat().st_size


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "LLM_VQC_weekly_benchmark_v2.pptx"
    )
    was, now = recompress(target)
    print(f"{target.name}: {was:,} -> {now:,} bytes ({100 * now / was:.0f}%)")
