#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-letter-semantic-sizes}"

run_one() {
    local levels="$1"
    local codebook="$2"

    local run_name="letter-l${levels}-c${codebook}"

    echo "Starting ${run_name} on GPU ${GPU}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    NUM_LEVELS="${levels}" \
    CODEBOOK_SIZE="${codebook}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" run.py experiments/letter_semantic_sizes.py \
        --skip-ranking \
        --mlflow \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
        --mlflow-run-name "${run_name}"
}


for levels in 2 3 4; do
    for codebook in 128 256 512; do
        run_one "${levels}" "${codebook}"
    done
done