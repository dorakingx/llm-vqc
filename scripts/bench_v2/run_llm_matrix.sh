#!/usr/bin/env bash
# All REAL-LLM bench_v2 cells, resumably, in dependency order, followed by
# the experiments that depend on them (E5 theta isolation, E4 robustness)
# and the per-experiment analyses.
#
# HARD PRECONDITIONS (refused otherwise, before any client is built):
#   OPENAI_API_KEY        - credential (from the repo-root .env)
#   OPENAI_MODEL          - explicit model id (no silent default)
#   LLM_API_BUDGET_USD    - positive dollar cap; the ONLY authorization
#                           for paid calls (a key alone is not one).
# Cost accounting is conservative: every call reserves $0.05 against the
# cap regardless of true cost, and each cell has a hard 40-call cap.
# The full LLM matrix (~220 cells, ~1500-2200 calls) therefore needs
# LLM_API_BUDGET_USD >= 120 to run uninterrupted, while the ACTUAL spend
# with a mini-class model is expected to be far lower (~$5-20).
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="/Users/hatanakatomoya/Developer/Sim/llm-vqc/.venv/bin/python"
[ -x "$PY" ] || PY=".venv/bin/python"
LOG_DIR="runs/bench_v2/logs"
mkdir -p "$LOG_DIR"

# Load the key/model from .env WITHOUT printing anything.
if [ -f .env ]; then set -a; source .env; set +a; fi

if [ -z "${LLM_API_BUDGET_USD:-}" ]; then
    echo "REFUSING: LLM_API_BUDGET_USD is not set. Export a positive cap first."
    exit 2
fi

run_with_retries() {
    local label="$1"; shift
    for attempt in 1 2 3; do
        echo "[llm-driver] $label attempt $attempt"
        if "$PY" "$@"; then echo "[llm-driver] $label OK"; return 0; fi
        echo "[llm-driver] $label attempt $attempt FAILED; retrying (cells resume)"
    done
    echo "[llm-driver] $label FAILED after 3 attempts"
    return 1
}

overall=0
run_with_retries "E1-llm" scripts/bench_v2/run_experiment.py \
    --experiment E1 --arms llm_open_joint llm_closed_joint || overall=1
run_with_retries "E2-llm" scripts/bench_v2/run_experiment.py \
    --experiment E2 --arms llm_open_structure llm_archive_closed_structure || overall=1
run_with_retries "E3-llm" scripts/bench_v2/run_experiment.py \
    --experiment E3 --arms llm_open_structure llm_archive_closed_structure || overall=1

# Dependent experiments + analyses (offline; no API calls).
run_with_retries "E5" scripts/bench_v2/run_e5_theta_isolation.py || overall=1
run_with_retries "E4" scripts/bench_v2/run_e4_robustness.py || overall=1
for eid in E1 E2 E3; do
    run_with_retries "analyze-$eid" scripts/bench_v2/analyze_experiment.py \
        --experiment "$eid" || overall=1
done

echo "[llm-driver] LLM matrix finished with status $overall"
exit $overall
