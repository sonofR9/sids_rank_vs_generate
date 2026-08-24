#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    printf 'Usage: %s GPU_ID\n' "$0" >&2
    exit 2
fi

GPU_ID="$1"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    printf 'GPU_ID must be a non-negative integer\n' >&2
    exit 2
fi

REPO_ROOT="${SID_REPO_ROOT:?SID_REPO_ROOT is required}"
PYTHON_BIN="${SID_PYTHON_BIN:?SID_PYTHON_BIN is required}"
REPORTS_ROOT="${SID_REPORTS_DIR:?SID_REPORTS_DIR is required}"
MLFLOW_URI="${SID_MLFLOW_URI:?SID_MLFLOW_URI is required}"
QUEUE_STATE_DIR="${SID_QUEUE_STATE_DIR:-$REPORTS_ROOT/.user-aware-quality-queue}"
MATRIX_STATE_DIR="${SID_MATRIX_STATE_DIR:-$REPORTS_ROOT/.user-aware-quality-state}"
MATRIX_LOG_DIR="${SID_MATRIX_LOG_DIR:-$MATRIX_STATE_DIR/logs}"
MANIFEST_PATH="$QUEUE_STATE_DIR/matrix-manifest.json"
EXPORT_PATH="$REPORTS_ROOT/user-aware-quality-results.partial.csv"
QUEUE_LOG_PATH="$QUEUE_STATE_DIR/queue.log"
STATUS_PATH="$QUEUE_STATE_DIR/status"
CURRENT_RUN_PATH="$QUEUE_STATE_DIR/current-run"
PID_PATH="$QUEUE_STATE_DIR/queue.pid"
LOCK_PATH="$QUEUE_STATE_DIR/gpu-$GPU_ID.lock"
BRIDGE_ROOT="$REPORTS_ROOT/.poster-bridge"

mkdir -p "$QUEUE_STATE_DIR" "$MATRIX_STATE_DIR" "$MATRIX_LOG_DIR"
cd "$REPO_ROOT"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    printf 'quality queue is already running for GPU %s\n' "$GPU_ID" >&2
    exit 3
fi

exec > >(tee -a "$QUEUE_LOG_PATH") 2>&1

set_status() {
    printf '%s\n' "$1" > "$STATUS_PATH"
}

set_current_run() {
    printf '%s\n' "$1" > "$CURRENT_RUN_PATH"
}

fail_queue() {
    status=$?
    set_status "FAILED exit=$status"
    printf 'queue failed exit=%s current=%s\n' \
        "$status" "$(cat "$CURRENT_RUN_PATH" 2>/dev/null || true)"
    exit "$status"
}

trap fail_queue ERR
printf '%s\n' "$$" > "$PID_PATH"
set_status "PREFLIGHT"
set_current_run "preflight"
printf 'queue started pid=%s gpu=%s repo=%s reports=%s\n' \
    "$$" "$GPU_ID" "$REPO_ROOT" "$REPORTS_ROOT"

test -x "$PYTHON_BIN"
if [[ -n "${SID_SOURCE_SHA256:-}" ]]; then
    SOURCE_SHA256="$SID_SOURCE_SHA256"
else
    test -f "$REPO_ROOT/SOURCE_SHA256"
    SOURCE_SHA256="$(tr -d '[:space:]' < "$REPO_ROOT/SOURCE_SHA256")"
fi
if [[ ! "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'SOURCE_SHA256 is invalid\n' >&2
    exit 2
fi
printf '%s\n' "$SOURCE_SHA256" > "$QUEUE_STATE_DIR/source.sha256"
printf 'source sha256=%s\n' "$SOURCE_SHA256"
command -v flock >/dev/null
command -v curl >/dev/null
test "$(df --output=avail -B1 "$REPORTS_ROOT" | tail -n 1)" -ge 107374182400

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"
name = torch.cuda.get_device_name(0)
assert "V100" in name, f"expected V100, received {name}"
free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
assert free_bytes >= 15 * 1024**3, "less than 15 GiB is free on the GPU"
print(f"cuda smoke ok: {name}, free={free_bytes / 1024**3:.2f} GiB")
PY

"$PYTHON_BIN" - <<'PY'
import mlflow
import numpy
import polars
import sklearn
import torch

print(
    "imports ok:",
    f"torch={torch.__version__}",
    f"polars={polars.__version__}",
    f"mlflow={mlflow.__version__}",
    f"sklearn={sklearn.__version__}",
    f"numpy={numpy.__version__}",
)
PY

curl --fail --silent --show-error "$MLFLOW_URI/health" >/dev/null

env \
    -u SID_REPO_ROOT \
    -u SID_PYTHON_BIN \
    -u SID_REPORTS_DIR \
    -u SID_MLFLOW_URI \
    -u SID_QUEUE_STATE_DIR \
    -u SID_MATRIX_STATE_DIR \
    -u SID_MATRIX_LOG_DIR \
    -u SID_CURRENT_LOG_PATH_FILE \
    PYTHONPATH=src \
    "$PYTHON_BIN" -m pytest -q \
    tests/test_poster_full_protocol.py \
    tests/test_sid_quality.py \
    tests/test_user_aware_quality_experiment.py \
    tests/test_user_aware_quality_matrix.py \
    tests/test_verify_poster_bridge.py \
    tests/test_user_aware_quality_results.py \
    tests/test_user_aware_quality_launcher.py

"$PYTHON_BIN" scripts/plan_user_aware_quality_matrix.py \
    --format json > "$MANIFEST_PATH"
"$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
runs = manifest["runs"]
names = [run["run_name"] for run in runs]
assert manifest["run_count"] == 30
assert len(names) == len(set(names)) == 30
print("matrix manifest ok: 30 unique runs")
PY

set_status "BRIDGE"
BRIDGE_ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
BRIDGE_REPORTS_DIR="$BRIDGE_ROOT/$BRIDGE_ATTEMPT"
BRIDGE_LOG_PATH="$QUEUE_STATE_DIR/bridge.$BRIDGE_ATTEMPT.log"
mkdir -p "$BRIDGE_REPORTS_DIR"
set_current_run "bridge-rqkmeans-l3-c256"
printf '%s\n' "$BRIDGE_LOG_PATH" > "$QUEUE_STATE_DIR/current-log.path"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
SID_REPORTS_DIR="$BRIDGE_REPORTS_DIR" \
PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" -u run.py \
    experiments/poster_full_33148_bridge.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "$MLFLOW_URI" \
    > "$BRIDGE_LOG_PATH" 2>&1

mapfile -t BRIDGE_METRICS < <(
    find "$BRIDGE_REPORTS_DIR" -type f -name metrics.json -print
)
if (( ${#BRIDGE_METRICS[@]} != 1 )); then
    printf 'bridge produced %s metrics files\n' "${#BRIDGE_METRICS[@]}" >&2
    exit 10
fi
"$PYTHON_BIN" scripts/verify_poster_bridge.py "${BRIDGE_METRICS[0]}"
printf '%s\n' "${BRIDGE_METRICS[0]}" > "$QUEUE_STATE_DIR/bridge-metrics.path"
printf 'bridge verified: %s\n' "${BRIDGE_METRICS[0]}"

set_status "MATRIX"

mapfile -t CANONICAL_RUNS < <(
    "$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
names = [run["run_name"] for run in manifest["runs"]]
priority = [
    "uaq30-v1-anchor-l3-c256",
    "uaq30-v2-fusion-weight-0",
    "uaq30-v2-anchor-l3-c256",
    "uaq30-v1-capacity-l4-c256",
    "uaq30-v2-capacity-l4-c256",
    "uaq30-v1-capacity-l3-c1024",
    "uaq30-v2-capacity-l3-c1024",
]
assert set(priority).issubset(names)
ordered = priority + [name for name in names if name not in priority]
assert len(ordered) == len(set(ordered)) == 30
print("\n".join(ordered))
PY
)

export SID_REPO_ROOT="$REPO_ROOT"
export SID_PYTHON_BIN="$PYTHON_BIN"
export SID_REPORTS_DIR="$REPORTS_ROOT"
export SID_MLFLOW_URI="$MLFLOW_URI"
export SID_MATRIX_STATE_DIR="$MATRIX_STATE_DIR"
export SID_MATRIX_LOG_DIR="$MATRIX_LOG_DIR"
export SID_CURRENT_LOG_PATH_FILE="$QUEUE_STATE_DIR/current-log.path"

for run_name in "${CANONICAL_RUNS[@]}"; do
    marker="$MATRIX_STATE_DIR/completed/$run_name.json"
    if [[ -f "$marker" ]]; then
        printf 'skip completed run=%s marker=%s\n' "$run_name" "$marker"
        continue
    fi

    set_current_run "$run_name"
    printf 'starting run=%s\n' "$run_name"
    scripts/run_user_aware_quality_point.sh "$run_name" "$GPU_ID"

    latest_log="$(
        find "$MATRIX_LOG_DIR" -maxdepth 1 -type f \
            -name "$run_name.*.log" -print | sort | tail -n 1
    )"
    printf '%s\n' "$latest_log" > "$QUEUE_STATE_DIR/current-log.path"

    "$PYTHON_BIN" scripts/export_user_aware_quality_results.py \
        --reports-root "$REPORTS_ROOT" \
        --output "$EXPORT_PATH" \
        --matrix-manifest "$MANIFEST_PATH"
    printf 'validated run=%s\n' "$run_name"
done

"$PYTHON_BIN" scripts/export_user_aware_quality_results.py \
    --reports-root "$REPORTS_ROOT" \
    --output "$REPORTS_ROOT/user-aware-quality-results.csv" \
    --matrix-manifest "$MANIFEST_PATH"

"$PYTHON_BIN" - "$REPORTS_ROOT/user-aware-quality-results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 30, f"expected 30 final rows, received {len(rows)}"
print("final export ok: 30 rows")
PY

set_current_run "none"
set_status "FINISHED"
printf 'queue finished successfully\n'
