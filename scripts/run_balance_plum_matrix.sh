#!/usr/bin/env bash
set -euo pipefail


RUN_SET="${RUN_SET:-all}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-test}"


run_balance() {
    local codebook="$1"
    local collision="$2"

    local run_name="poster-balance-${collision}-l3-c${codebook}-full-seed42"

    echo "============================================================"
    echo "Starting ${run_name}"
    echo "GPU=${GPU}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    SID_MODEL="balance" \
    COLLISION_RESOLUTION="${collision}" \
    NUM_LEVELS="3" \
    CODEBOOK_SIZE="${codebook}" \
    "${PYTHON_BIN}" run.py experiments/balance_plum_poster.py \
        --mlflow \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
        --mlflow-run-name "${run_name}"
}


run_plum() {
    local equivalent_codebook="$1"
    local codebook_sizes="$2"
    local collision="$3"

    local sizes_name
    sizes_name="${codebook_sizes//,/x}"

    local run_name="poster-plum-${collision}-l3-c${equivalent_codebook}eq-cbs${sizes_name}-full-seed42"

    echo "============================================================"
    echo "Starting ${run_name}"
    echo "GPU=${GPU}"
    echo "CODEBOOK_SIZES=${codebook_sizes}"
    echo "Equivalent uniform codebook: C=${equivalent_codebook}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    SID_MODEL="plum" \
    COLLISION_RESOLUTION="${collision}" \
    CODEBOOK_SIZES="${codebook_sizes}" \
    "${PYTHON_BIN}" run.py experiments/balance_plum_poster.py \
        --skip-ranking \
        --mlflow \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
        --mlflow-run-name "${run_name}"
}


balance_128() {
    run_balance 128 none
    run_balance 128 token
}


balance_256() {
    run_balance 256 none
    run_balance 256 token
}


balance_512() {
    run_balance 512 none
    run_balance 512 token
}


plum_128() {
    # Same SID space size as L=3, C=128:
    # 256 * 128 * 64 = 128^3
    run_plum 128 "256,128,64" none
    run_plum 128 "256,128,64" token
}


plum_256() {
    # Same SID space size as L=3, C=256:
    # 512 * 256 * 128 = 256^3
    run_plum 256 "512,256,128" none
    run_plum 256 "512,256,128" token
}


plum_512() {
    # Same SID space size as L=3, C=512:
    # 1024 * 512 * 256 = 512^3
    run_plum 512 "1024,512,256" none
    run_plum 512 "1024,512,256" token
}


balance_all() {
    balance_128
    balance_256
    balance_512
}


plum_all() {
    plum_128
    plum_256
    plum_512
}


case "${RUN_SET}" in

    balance_128)
        balance_128
        ;;

    balance_256)
        balance_256
        ;;

    balance_512)
        balance_512
        ;;

    balance_all)
        balance_all
        ;;

    plum_128)
        plum_128
        ;;

    plum_256)
        plum_256
        ;;

    plum_512)
        plum_512
        ;;

    plum_all)
        plum_all
        ;;

    all)
        balance_all
        plum_all
        ;;

    *)
        echo "Unknown RUN_SET: ${RUN_SET}" >&2
        echo "Available:" >&2
        echo "  balance_128" >&2
        echo "  balance_256" >&2
        echo "  balance_512" >&2
        echo "  balance_all" >&2
        echo "  plum_128" >&2
        echo "  plum_256" >&2
        echo "  plum_512" >&2
        echo "  plum_all" >&2
        echo "  all" >&2
        exit 2
        ;;
esac