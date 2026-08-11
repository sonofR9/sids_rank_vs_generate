#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-levels}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-logq-rqvae-generation}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/logq_rqvae_runs/_cache}"
LOGQ_MODE="${LOGQ_MODE:-corrected}"

run_one() {
  local levels="$1"
  local codebook="$2"
  local run_name="logq-rqvae-rqkmeans-nearest-token-l${levels}-c${codebook}-aux0p1-${LOGQ_MODE}-full-seed42"

  echo "Starting ${run_name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  LOGQ_MODE="${LOGQ_MODE}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/logq_poster_rqvae.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

levels() {
  run_one 2 256
  run_one 3 256
  run_one 4 256
}

codebooks() {
  run_one 3 128
  run_one 3 512
  run_one 3 1024
}

case "${RUN_SET}" in
  levels) levels ;;
  codebooks) codebooks ;;
  all) levels; codebooks ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}; use levels, codebooks, or all" >&2
    exit 2
    ;;
esac
