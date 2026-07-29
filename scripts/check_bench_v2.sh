#!/usr/bin/env bash
# bench_v2 CI gate: full test suite + lint on the bench_v2 surface.
#
# Lint scope note (recorded in DECISIONS.md, 2026-07-29): the preserved
# experiment lineage carries ~965 pre-existing ruff findings that appeared
# through ruff version drift (0.6-era code, 0.15.x in the venv). Those
# files are frozen provenance for committed artifacts and are NOT
# reformatted; every bench_v2 path below must be clean instead.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${VIRTUAL_ENV:-}" ]; then PYTHON="python";
elif [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python";
elif [ -x "../../../.venv/bin/python" ]; then PYTHON="../../../.venv/bin/python";
else PYTHON="python3"; fi

echo "==> ruff check (bench_v2 surface)"
"$PYTHON" -m ruff check \
    llm_vqc/tasks/signal_suite \
    llm_vqc/bench_v2 \
    tests/test_signal_suite.py \
    tests/test_signal_suite_baselines.py \
    tests/test_bench_v2*.py \
    scripts/check_goal_completion.py \
    scripts/bench_v2 \
    2>/dev/null || {
        # Re-run without the not-yet-existing paths silenced so real
        # failures are visible.
        existing=()
        for p in llm_vqc/tasks/signal_suite llm_vqc/bench_v2 \
                 tests/test_signal_suite.py tests/test_signal_suite_baselines.py \
                 scripts/check_goal_completion.py scripts/bench_v2; do
            [ -e "$p" ] && existing+=("$p")
        done
        shopt -s nullglob
        for f in tests/test_bench_v2*.py; do existing+=("$f"); done
        shopt -u nullglob
        "$PYTHON" -m ruff check "${existing[@]}"
    }

echo "==> pytest (full suite)"
"$PYTHON" -m pytest tests/ -q

echo "==> bench_v2 checks passed"
