#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-letter-semantic-sizes}"


run_one() {
    local gpu="$1"
    local levels="$2"
    local codebook="$3"

    local run_name="letter-l${levels}-c${codebook}"

    echo "Starting ${run_name} on GPU ${gpu}"

    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        NUM_LEVELS="${levels}" \
        CODEBOOK_SIZE="${codebook}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" run.py experiments/rqvae_letter_full.py \
            --skip-ranking \
            --mlflow \
            --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
            --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
            --mlflow-run-name "${run_name}"
    )

    echo "Finished ${run_name} on GPU ${gpu}"
}


# GPU 2 — 5 экспериментов последовательно
(
    run_one 2 2 128
    run_one 2 2 256
    run_one 2 2 512
    run_one 2 3 128
    run_one 2 3 256
) &
PID_GPU2=$!


# GPU 3 — 4 эксперимента последовательно
(
    run_one 3 3 512
    run_one 3 4 128
    run_one 3 4 256
    run_one 3 4 512
) &
PID_GPU3=$!


# Ждём обе очереди
STATUS=0

wait "${PID_GPU2}" || STATUS=$?
wait "${PID_GPU3}" || STATUS=$?

if [[ "${STATUS}" -ne 0 ]]; then
    echo "One of the experiment queues failed" >&2
    exit "${STATUS}"
fi
