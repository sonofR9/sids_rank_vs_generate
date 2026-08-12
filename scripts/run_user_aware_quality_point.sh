#!/usr/bin/env bash

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    printf 'Usage: %s MATRIX_RUN_NAME [GPU_ID]\n' "$0" >&2
    exit 2
fi

RUN_NAME="$1"
GPU_ID="${2:-${SID_GPU:-}}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    printf 'GPU_ID is required and must be a non-negative integer\n' >&2
    exit 2
fi
REPO_ROOT="${SID_REPO_ROOT:-$PWD}"
PYTHON_BIN="${SID_PYTHON_BIN:-python}"
REPORTS_DIR="${SID_REPORTS_DIR:-$REPO_ROOT/reports/experiments}"
MLFLOW_URI="${SID_MLFLOW_URI:-http://51.250.47.231:5000}"
STATE_DIR="${SID_MATRIX_STATE_DIR:-$REPORTS_DIR/.user-aware-quality-state}"
COMPLETION_DIR="$STATE_DIR/completed"
LOCK_DIR="$STATE_DIR/locks"
LOG_DIR="${SID_MATRIX_LOG_DIR:-$STATE_DIR/logs}"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/plan_user_aware_quality_matrix.py \
    --run-name "$RUN_NAME" \
    --format table

command -v flock >/dev/null || {
    printf 'flock is required for a nonblocking per-run lock\n' >&2
    exit 2
}

mkdir -p "$COMPLETION_DIR" "$LOCK_DIR" "$LOG_DIR"
exec 9>"$LOCK_DIR/$RUN_NAME.lock"
if ! flock -n 9; then
    printf 'matrix run is already locked: %s\n' "$RUN_NAME" >&2
    exit 3
fi

COMPLETION_MARKER="$COMPLETION_DIR/$RUN_NAME.json"
if [[ -e "$COMPLETION_MARKER" ]]; then
    printf 'matrix run already completed: %s\n' "$COMPLETION_MARKER" >&2
    exit 4
fi

ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEMPT_REPORTS_DIR="$REPORTS_DIR/.user-aware-quality-runs/$RUN_NAME/$ATTEMPT_ID"
LOG_PATH="$LOG_DIR/$RUN_NAME.$ATTEMPT_ID.log"
mkdir -p "$ATTEMPT_REPORTS_DIR"
if [[ -n "${SID_CURRENT_LOG_PATH_FILE:-}" ]]; then
    mkdir -p "$(dirname "$SID_CURRENT_LOG_PATH_FILE")"
    printf '%s\n' "$LOG_PATH" > "$SID_CURRENT_LOG_PATH_FILE"
fi
if [[ -n "${SID_PUBLIC_CURRENT_LOG_PATH_FILE:-}" ]]; then
    mkdir -p "$(dirname "$SID_PUBLIC_CURRENT_LOG_PATH_FILE")"
    printf '%s\n' "$LOG_PATH" > "$SID_PUBLIC_CURRENT_LOG_PATH_FILE"
fi

printf 'run=%s gpu=%s attempt=%s log=%s\n' \
    "$RUN_NAME" "$GPU_ID" "$ATTEMPT_ID" "$LOG_PATH"

if env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    SID_MATRIX_RUN="$RUN_NAME" \
    SID_REPORTS_DIR="$ATTEMPT_REPORTS_DIR" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u run.py \
    experiments/user_aware_quality_matrix_run.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "$MLFLOW_URI" \
    >"$LOG_PATH" 2>&1
then
    if [[ -n "${SID_GLOBAL_SID_CANONICAL_SHA256:-}" ]]; then
        mapfile -t POINT_METRICS < <(
            find "$ATTEMPT_REPORTS_DIR" -type f -name metrics.json -print
        )
        if (( ${#POINT_METRICS[@]} != 1 )); then
            printf 'matrix point produced %s metrics files\n' \
                "${#POINT_METRICS[@]}" >&2
            exit 10
        fi
        "$PYTHON_BIN" scripts/verify_global_sid_anchor.py \
            "${POINT_METRICS[0]}" \
            --expected-sha256 "$SID_GLOBAL_SID_CANONICAL_SHA256" \
            --profile v100-fp32-20260812
    fi
    "$PYTHON_BIN" scripts/finalize_user_aware_quality_result.py \
        --reports-root "$REPORTS_DIR" \
        --attempt-reports-dir "$ATTEMPT_REPORTS_DIR" \
        --completion-dir "$COMPLETION_DIR" \
        --run-name "$RUN_NAME" \
        --attempt-id "$ATTEMPT_ID" \
        --gpu-id "$GPU_ID" \
        --log-path "$LOG_PATH"
    printf 'matrix run completed: %s\n' "$RUN_NAME"
else
    status=$?
    printf 'matrix run failed with exit %s; log=%s\n' \
        "$status" "$LOG_PATH" >&2
    exit "$status"
fi
