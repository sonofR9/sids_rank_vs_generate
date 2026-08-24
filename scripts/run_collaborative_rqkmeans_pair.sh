#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-pair}"
ALIGNED_GPU="${ALIGNED_GPU:-0}"
BASELINE_GPU="${BASELINE_GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"
REPORTS_ROOT="${REPORTS_ROOT:-reports/poster_one_rec_pair_comparison}"

aligned_run_name="poster-onerec-pair-alignment-l3-c256-full-seed42-catalog33148"
baseline_run_name="poster-rqkmeans-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148"

preflight() {
  CUDA_VISIBLE_DEVICES="${ALIGNED_GPU},${BASELINE_GPU}" \
  "${PYTHON_BIN}" - <<'PY'
import torch
from sids_rank_vs_generate.datasets import resolve_dataset

if torch.cuda.device_count() != 2:
    raise SystemExit(
        "Expected exactly two selected GPUs, received "
        f"{torch.cuda.device_count()}"
    )
print(f"Selected GPU 0: {torch.cuda.get_device_name(0)}")
print(f"Selected GPU 1: {torch.cuda.get_device_name(1)}")
files = resolve_dataset("50m")
print(f"Prepared dataset: {files.interactions_path}")
print(f"Prepared embeddings: {files.embeddings_path}")
PY
}

run_aligned() {
  echo "Starting ${aligned_run_name} on physical GPU ${ALIGNED_GPU}"
  CUDA_VISIBLE_DEVICES="${ALIGNED_GPU}" \
  SID_REPORTS_DIR="${REPORTS_ROOT}/aligned" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/collaborative_rqkmeans_poster_full.py \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${aligned_run_name}"
}

run_baseline() {
  echo "Starting ${baseline_run_name} on physical GPU ${BASELINE_GPU}"
  CUDA_VISIBLE_DEVICES="${BASELINE_GPU}" \
  SID_REPORTS_DIR="${REPORTS_ROOT}/baseline" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/poster_full_33148_bridge.py \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${baseline_run_name}"
}

preflight

case "${RUN_SET}" in
  aligned)
    run_aligned
    ;;
  baseline)
    run_baseline
    ;;
  pair)
    run_aligned &
    aligned_pid=$!
    run_baseline &
    baseline_pid=$!

    task_exit=0
    wait "${aligned_pid}" || task_exit=$?
    wait "${baseline_pid}" || task_exit=$?
    exit "${task_exit}"
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}; use aligned, baseline, or pair" >&2
    exit 2
    ;;
esac
