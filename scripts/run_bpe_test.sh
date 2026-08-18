#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-bpe}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

RUN_SET="${RUN_SET:-all}"

run_one() {
  local gpu="$1"
  local levels="$2"
  local codebook="$3"
  local merges="$4"

  local run_name="bpe-l${levels}-c${codebook}-m${merges}"

  echo "========================================"
  echo "Starting ${run_name}"
  echo "GPU=${gpu}"
  echo "L=${levels}, C=${codebook}, M=${merges}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  ASSIGNMENT_METHOD="nearest" \
  COLLISION_RESOLUTION="none" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  NUM_MERGES="${merges}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/bpe_full.py \
      --skip-ranking \
      --mlflow \
      --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
      --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
      --mlflow-run-name "${run_name}"
}

case "${RUN_SET}" in
  l3-c128)
    run_one 2 3 128 512
    ;;

  l5-c128)
    run_one 2 5 128 512
    ;;

  l5-c64)
    run_one 3 5 64 512
    ;;

  l7-c64)
    run_one 3 7 64 512
    ;;

  all)
    (
      run_one 2 3 128 512
      run_one 2 5 128 512
    ) &

    gpu2_pid=$!

    (
      run_one 3 5 64 512
      run_one 3 7 64 512
    ) &

    gpu3_pid=$!

    wait "${gpu2_pid}"
    wait "${gpu3_pid}"
    ;;

  *)
    echo "Unknown RUN_SET=${RUN_SET}" >&2
    echo "Use: l3-c128, l5-c128, l5-c64, l7-c64, or all" >&2
    exit 2
    ;;
esac