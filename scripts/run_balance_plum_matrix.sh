#!/usr/bin/env bash
set -euo pipefail


RUN_SET="${RUN_SET:-ranking_none_all}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-test}"


run_balance() {
    local codebook="$1"
    local run_name="poster-ranking-balance-none-l3-c${codebook}-full-seed42"

    echo "Starting ${run_name} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    SID_MODEL="balance" \
    COLLISION_RESOLUTION="none" \
    NUM_LEVELS="3" \
    CODEBOOK_SIZE="${codebook}" \
    "${PYTHON_BIN}" run.py experiments/balance_plum_poster.py \
        --skip-generation \
        --mlflow \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
        --mlflow-run-name "${run_name}"
}


run_plum() {
    local equivalent_codebook="$1"
    local codebook_sizes="$2"
    local sizes_name="${codebook_sizes//,/x}"
    local run_name="poster-ranking-plum-none-l3-c${equivalent_codebook}eq-cbs${sizes_name}-full-seed42"

    echo "Starting ${run_name} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    SID_MODEL="plum" \
    COLLISION_RESOLUTION="none" \
    CODEBOOK_SIZES="${codebook_sizes}" \
    "${PYTHON_BIN}" run.py experiments/balance_plum_poster.py \
        --skip-generation \
        --mlflow \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
        --mlflow-run-name "${run_name}"
}


balance_none_all() {
    run_balance 128
    run_balance 256
    run_balance 512
}


plum_none_all() {
    run_plum 128 "256,128,64"
    run_plum 256 "512,256,128"
    run_plum 512 "1024,512,256"
}


case "${RUN_SET}" in
    balance_128) run_balance 128 ;;
    balance_256) run_balance 256 ;;
    balance_512) run_balance 512 ;;
    balance_none_all) balance_none_all ;;
    plum_128) run_plum 128 "256,128,64" ;;
    plum_256) run_plum 256 "512,256,128" ;;
    plum_512) run_plum 512 "1024,512,256" ;;
    plum_none_all) plum_none_all ;;
    ranking_none_all)
        balance_none_all
        plum_none_all
        ;;
    *)
        echo "Unknown RUN_SET: ${RUN_SET}" >&2
        echo "Available: balance_128 balance_256 balance_512 balance_none_all" >&2
        echo "           plum_128 plum_256 plum_512 plum_none_all ranking_none_all" >&2
        exit 2
        ;;
esac
