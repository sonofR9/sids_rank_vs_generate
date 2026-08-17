#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-all}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-letter-semantic-sizes}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/letter_cache}"

run_one() {
    local levels="$1"
    local codebook="$2"

    local run_name="letter-l${levels}-c${codebook}"

    echo "Starting ${run_name}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
    SID_NUM_LEVELS="${levels}" \
    SID_CODEBOOK_SIZE="${codebook}" \
    RUN_NAME="${run_name}" \
    "${PYTHON_BIN}" run.py
}

case "${RUN_SET}" in
    all)
        run_one 2 256
        run_one 2 512
        run_one 2 1024

        run_one 3 256
        run_one 3 512

        run_one 4 256
        ;;

    *)
        echo "Unknown RUN_SET=${RUN_SET}"
        exit 1
        ;;
esac