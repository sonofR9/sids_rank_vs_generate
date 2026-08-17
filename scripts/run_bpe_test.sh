#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

LEVELS=3
CODEBOOK=256
NUM_MERGES=512

RUN_NAME="bpe-l${LEVELS}-c${CODEBOOK}-m${NUM_MERGES}"

echo "========================================"
echo "Starting ${RUN_NAME}"
echo "GPU=${GPU}"
echo "L=${LEVELS}, C=${CODEBOOK}, merges=${NUM_MERGES}"
echo "========================================"

CUDA_VISIBLE_DEVICES="${GPU}" \
SID_MODEL="bpe" \
R3_INIT="rqkmeans" \
ASSIGNMENT_METHOD="nearest" \
COLLISION_RESOLUTION="none" \
NUM_LEVELS="${LEVELS}" \
CODEBOOK_SIZE="${CODEBOOK}" \
NUM_MERGES="${NUM_MERGES}" \
MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
"${PYTHON_BIN}" run.py experiments/bpe_full.py \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${RUN_NAME}"