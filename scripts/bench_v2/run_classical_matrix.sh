#!/usr/bin/env bash
# Full CLASSICAL (non-LLM) bench_v2 matrix, resumably, in dependency
# order: E2 (classical arms + references, 10 replicates) -> E3 (classical
# arms + resolved best reference, 5 replicates). Each experiment retries
# up to 3 times (cells are idempotent and resume from their stores), so a
# transient crash cannot silently lose a cell. LLM arms are intentionally
# absent here: scientific LLM cells require the real provider and are run
# by run_llm_matrix.sh under an explicit LLM_API_BUDGET_USD cap.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="/Users/hatanakatomoya/Developer/Sim/llm-vqc/.venv/bin/python"
[ -x "$PY" ] || PY=".venv/bin/python"
LOG_DIR="runs/bench_v2/logs"
mkdir -p "$LOG_DIR"

run_with_retries() {
    local label="$1"; shift
    for attempt in 1 2 3; do
        echo "[driver] $label attempt $attempt: $*"
        if "$PY" "$@"; then
            echo "[driver] $label OK"
            return 0
        fi
        echo "[driver] $label attempt $attempt FAILED; retrying (cells resume)"
    done
    echo "[driver] $label FAILED after 3 attempts"
    return 1
}

overall=0

run_with_retries "E2-classical" scripts/bench_v2/run_experiment.py \
    --experiment E2 \
    --arms random_structure evolutionary_structure greedy_growth \
           ref_realamp_d1 ref_realamp_d2 ref_strongent_d1 ref_strongent_d2 \
    || overall=1

run_with_retries "E3-classical" scripts/bench_v2/run_experiment.py \
    --experiment E3 \
    --arms random_structure evolutionary_structure ref_best_validation \
    || overall=1

echo "[driver] classical matrix finished with status $overall"
exit $overall
