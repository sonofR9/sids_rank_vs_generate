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
MATRIX_STATE_DIR="${SID_MATRIX_STATE_DIR:?SID_MATRIX_STATE_DIR is required}"
BRIDGE_METRICS_PATH="${SID_PREVERIFIED_BRIDGE_METRICS_PATH:?SID_PREVERIFIED_BRIDGE_METRICS_PATH is required}"
BRIDGE_METRICS_SHA256="${SID_PREVERIFIED_BRIDGE_METRICS_SHA256:?SID_PREVERIFIED_BRIDGE_METRICS_SHA256 is required}"
BRIDGE_SIDS_PATH="${SID_PREVERIFIED_BRIDGE_SIDS_PATH:?SID_PREVERIFIED_BRIDGE_SIDS_PATH is required}"
BRIDGE_SIDS_SHA256="${SID_PREVERIFIED_BRIDGE_SIDS_SHA256:?SID_PREVERIFIED_BRIDGE_SIDS_SHA256 is required}"
GLOBAL_SID_CANONICAL_SHA256="${SID_GLOBAL_SID_CANONICAL_SHA256:?SID_GLOBAL_SID_CANONICAL_SHA256 is required}"

CURRENT_VERIFIER_PROFILE="v100-fp32-20260812"
MATRIX_ID="user-aware-quality-30-current-anchor-v1"
GLOBAL_SID_ANCHOR_ID="current-rqkmeans-l3-c256-seed42-v1"
RESUME_STATE_DIR="${SID_RESUME_STATE_DIR:-$MATRIX_STATE_DIR/resume-current-anchor/gpu-$GPU_ID}"
MATRIX_LOG_DIR="${SID_MATRIX_LOG_DIR:-$MATRIX_STATE_DIR/logs}"
COMPLETION_DIR="$MATRIX_STATE_DIR/completed"
PUBLIC_STATE_DIR="${SID_PUBLIC_STATE_DIR:-$RESUME_STATE_DIR}"
LOCK_DIR="$MATRIX_STATE_DIR/queue-locks"
LOCK_PATH="$LOCK_DIR/gpu-$GPU_ID.quality-matrix.lock"
MANIFEST_PATH="$RESUME_STATE_DIR/matrix-manifest.json"
ACCEPTANCE_PATH="$RESUME_STATE_DIR/preverified-anchor.acceptance.json"
VERIFIER_OUTPUT_PATH="$RESUME_STATE_DIR/preverified-anchor.verifier.json"
PARTIAL_EXPORT_PATH="${SID_MATRIX_PARTIAL_EXPORT_PATH:-$REPORTS_ROOT/user-aware-quality-results.current-anchor.partial.csv}"
FINAL_EXPORT_PATH="${SID_MATRIX_FINAL_EXPORT_PATH:-$REPORTS_ROOT/user-aware-quality-results.current-anchor.final.csv}"
QUEUE_LOG_PATH="$RESUME_STATE_DIR/queue.log"
STATUS_PATH="$RESUME_STATE_DIR/status"
CURRENT_RUN_PATH="$RESUME_STATE_DIR/current-run"
CURRENT_LOG_PATH="$RESUME_STATE_DIR/current-log.path"
PID_PATH="$RESUME_STATE_DIR/queue.pid"
PUBLIC_STATUS_PATH="$PUBLIC_STATE_DIR/status"
PUBLIC_CURRENT_RUN_PATH="$PUBLIC_STATE_DIR/current-run"
PUBLIC_CURRENT_LOG_PATH="$PUBLIC_STATE_DIR/current-log.path"
PUBLIC_PID_PATH="$PUBLIC_STATE_DIR/queue.pid"

PLAN_SCRIPT="${SID_MATRIX_PLAN_SCRIPT:-$REPO_ROOT/scripts/plan_user_aware_quality_matrix.py}"
VERIFY_SCRIPT="${SID_BRIDGE_VERIFY_SCRIPT:-$REPO_ROOT/scripts/verify_poster_bridge.py}"
EXPORT_SCRIPT="${SID_MATRIX_EXPORT_SCRIPT:-$REPO_ROOT/scripts/export_user_aware_quality_results.py}"
POINT_RUNNER="${SID_MATRIX_POINT_RUNNER:-$REPO_ROOT/scripts/run_user_aware_quality_point.sh}"

for path in \
    "$REPO_ROOT" \
    "$REPORTS_ROOT" \
    "$MATRIX_STATE_DIR" \
    "$BRIDGE_METRICS_PATH" \
    "$BRIDGE_SIDS_PATH" \
    "$PARTIAL_EXPORT_PATH" \
    "$FINAL_EXPORT_PATH" \
    "$PUBLIC_STATE_DIR"
do
    if [[ "$path" != /* ]]; then
        printf 'all configured paths must be absolute: %s\n' "$path" >&2
        exit 2
    fi
done

if [[ "$REPORTS_ROOT" == "$MATRIX_STATE_DIR" ]]; then
    printf 'SID_REPORTS_DIR and SID_MATRIX_STATE_DIR must be separate\n' >&2
    exit 2
fi
if [[ "$PARTIAL_EXPORT_PATH" == "$FINAL_EXPORT_PATH" ]]; then
    printf 'partial and final export paths must be separate\n' >&2
    exit 2
fi
for digest in \
    "$BRIDGE_METRICS_SHA256" \
    "$BRIDGE_SIDS_SHA256" \
    "$GLOBAL_SID_CANONICAL_SHA256"
do
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'expected SHA256 values must be 64 lowercase hex characters\n' >&2
        exit 2
    fi
done

test -d "$REPO_ROOT"
test -x "$PYTHON_BIN"
test -s "$BRIDGE_METRICS_PATH"
test -s "$BRIDGE_SIDS_PATH"
if [[ -z "${SID_SOURCE_SHA256:-}" ]]; then
    test -f "$REPO_ROOT/SOURCE_SHA256"
fi
test -f "$PLAN_SCRIPT"
test -f "$VERIFY_SCRIPT"
test -f "$EXPORT_SCRIPT"
test -x "$POINT_RUNNER"
command -v flock >/dev/null

mkdir -p \
    "$REPORTS_ROOT" \
    "$RESUME_STATE_DIR" \
    "$MATRIX_LOG_DIR" \
    "$COMPLETION_DIR" \
    "$LOCK_DIR" \
    "$PUBLIC_STATE_DIR" \
    "$(dirname "$PARTIAL_EXPORT_PATH")" \
    "$(dirname "$FINAL_EXPORT_PATH")"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    printf 'quality matrix resume is already running on GPU %s\n' "$GPU_ID" >&2
    exit 3
fi

exec >> "$QUEUE_LOG_PATH" 2>&1

atomic_text() {
    local destination="$1"
    local value="$2"
    local temporary="${destination}.tmp.$$"
    printf '%s\n' "$value" > "$temporary"
    mv -f -- "$temporary" "$destination"
}

set_status() {
    atomic_text "$STATUS_PATH" "$1"
    if [[ "$PUBLIC_STATUS_PATH" != "$STATUS_PATH" ]]; then
        atomic_text "$PUBLIC_STATUS_PATH" "$1"
    fi
}

set_current_run() {
    atomic_text "$CURRENT_RUN_PATH" "$1"
    if [[ "$PUBLIC_CURRENT_RUN_PATH" != "$CURRENT_RUN_PATH" ]]; then
        atomic_text "$PUBLIC_CURRENT_RUN_PATH" "$1"
    fi
}

RUNNER_FINISHED=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 && RUNNER_FINISHED == 0 )); then
        local current="unknown"
        current="$(cat "$CURRENT_RUN_PATH" 2>/dev/null || true)"
        set_status "FAILED exit=$status current=$current"
        printf 'quality matrix resume failed exit=%s current=%s\n' \
            "$status" "$current" >&2
    fi
    exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

atomic_text "$PID_PATH" "$$"
if [[ "$PUBLIC_PID_PATH" != "$PID_PATH" ]]; then
    atomic_text "$PUBLIC_PID_PATH" "$$"
fi
set_current_run "preflight"
set_status "PREFLIGHT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

printf 'resume started pid=%s gpu=%s\n' "$$" "$GPU_ID"
printf 'local reports=%s\n' "$REPORTS_ROOT"
printf 'shared state=%s\n' "$MATRIX_STATE_DIR"
printf 'PYTHONPATH=%s\n' "$PYTHONPATH"

if [[ -n "${SID_SOURCE_SHA256:-}" ]]; then
    SOURCE_SHA256="$SID_SOURCE_SHA256"
else
    SOURCE_SHA256="$(tr -d '[:space:]' < "$REPO_ROOT/SOURCE_SHA256")"
fi
if [[ ! "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'SOURCE_SHA256 is invalid\n' >&2
    exit 2
fi
atomic_text "$RESUME_STATE_DIR/source.sha256" "$SOURCE_SHA256"
if [[ "$PUBLIC_STATE_DIR" != "$RESUME_STATE_DIR" ]]; then
    atomic_text "$PUBLIC_STATE_DIR/source.sha256" "$SOURCE_SHA256"
fi
printf 'source sha256=%s\n' "$SOURCE_SHA256"

sha256_file() {
    "$PYTHON_BIN" -c \
        'import hashlib, pathlib, sys; p = pathlib.Path(sys.argv[1]); h = hashlib.sha256(); f = p.open("rb"); [h.update(chunk) for chunk in iter(lambda: f.read(1048576), b"")]; f.close(); print(h.hexdigest())' \
        "$1"
}

set_current_run "preverified-anchor"
set_status "VERIFY_PREVERIFIED_ANCHOR"

ACTUAL_METRICS_SHA256="$(sha256_file "$BRIDGE_METRICS_PATH")"
if [[ "$ACTUAL_METRICS_SHA256" != "$BRIDGE_METRICS_SHA256" ]]; then
    printf 'preverified bridge metrics SHA256 mismatch\n' >&2
    exit 11
fi

ACTUAL_SIDS_SHA256="$(sha256_file "$BRIDGE_SIDS_PATH")"
if [[ "$ACTUAL_SIDS_SHA256" != "$BRIDGE_SIDS_SHA256" ]]; then
    printf 'preverified bridge semantic IDs SHA256 mismatch\n' >&2
    exit 12
fi

ACTUAL_CANONICAL_SHA256="$(
    "$PYTHON_BIN" -c '
import hashlib
import pathlib
import sys

import numpy as np
import polars as pl

columns = ["item_id", "sid_0", "sid_1", "sid_2"]
frame = pl.read_parquet(pathlib.Path(sys.argv[1])).select(columns).sort("item_id")
frame = frame.with_columns(pl.col(column).cast(pl.Int64) for column in columns)
if frame.height != 33_148:
    raise SystemExit(f"expected 33148 global SID rows, received {frame.height}")
if frame.null_count().to_numpy().sum() != 0:
    raise SystemExit("global SID anchor contains nulls")
if frame.get_column("item_id").n_unique() != frame.height:
    raise SystemExit("global SID anchor contains duplicate item_id values")
matrix = np.ascontiguousarray(frame.to_numpy(), dtype=np.int64)
print(hashlib.sha256(matrix.tobytes(order="C")).hexdigest())
' "$BRIDGE_SIDS_PATH"
)"
if [[ "$ACTUAL_CANONICAL_SHA256" != "$GLOBAL_SID_CANONICAL_SHA256" ]]; then
    printf 'preverified bridge canonical global SID SHA256 mismatch\n' >&2
    exit 13
fi

"$PYTHON_BIN" "$VERIFY_SCRIPT" \
    "$BRIDGE_METRICS_PATH" \
    --profile "$CURRENT_VERIFIER_PROFILE" \
    > "$VERIFIER_OUTPUT_PATH"

MANIFEST_TEMP="$RESUME_STATE_DIR/matrix-manifest.tmp.$$"
"$PYTHON_BIN" "$PLAN_SCRIPT" --format json > "$MANIFEST_TEMP"
"$PYTHON_BIN" - "$MANIFEST_TEMP" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    manifest = json.load(stream)
runs = manifest["runs"]
names = [run["run_name"] for run in runs]
assert manifest["matrix_id"] == "user-aware-quality-30-current-anchor-v1"
assert manifest["run_count"] == 30
assert len(names) == len(set(names)) == 30
assert manifest["protocol"] == {
    "expected_catalog_size": 33_148,
    "id": "poster-full-33148-v1",
}
assert manifest["fixed_global_sid"]["anchor_id"] == (
    "current-rqkmeans-l3-c256-seed42-v1"
)
assert manifest["fixed_global_sid"]["reference_scope"] == "current_anchor"
print("matrix manifest ok: 30 unique current-anchor runs")
PY
mv -f -- "$MANIFEST_TEMP" "$MANIFEST_PATH"
MANIFEST_SHA256="$(sha256_file "$MANIFEST_PATH")"

"$PYTHON_BIN" - \
    "$ACCEPTANCE_PATH" \
    "$VERIFIER_OUTPUT_PATH" \
    "$MATRIX_ID" \
    "$GLOBAL_SID_ANCHOR_ID" \
    "$CURRENT_VERIFIER_PROFILE" \
    "$BRIDGE_METRICS_PATH" \
    "$ACTUAL_METRICS_SHA256" \
    "$BRIDGE_SIDS_PATH" \
    "$ACTUAL_SIDS_SHA256" \
    "$ACTUAL_CANONICAL_SHA256" \
    "$MANIFEST_PATH" \
    "$MANIFEST_SHA256" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone

(
    output_text,
    verifier_text,
    matrix_id,
    anchor_id,
    verifier_profile,
    metrics_text,
    metrics_sha256,
    semantic_ids_text,
    semantic_ids_sha256,
    canonical_sha256,
    matrix_manifest_text,
    matrix_manifest_sha256,
) = sys.argv[1:]
output = pathlib.Path(output_text)
verifier_output = pathlib.Path(verifier_text)
verifier = json.loads(verifier_output.read_text(encoding="utf-8"))
if verifier.get("reference_profile") != verifier_profile:
    raise SystemExit("bridge verifier did not report the required current profile")
payload = {
    "schema_version": 1,
    "status": "ACCEPTED",
    "accepted_at": datetime.now(timezone.utc).isoformat(),
    "matrix_id": matrix_id,
    "global_sid_anchor_id": anchor_id,
    "verifier_profile": verifier_profile,
    "bridge_metrics_path": str(pathlib.Path(metrics_text).resolve()),
    "bridge_metrics_sha256": metrics_sha256,
    "bridge_semantic_ids_path": str(pathlib.Path(semantic_ids_text).resolve()),
    "bridge_semantic_ids_sha256": semantic_ids_sha256,
    "global_sid_canonical_sha256": canonical_sha256,
    "global_sid_canonical_algorithm": (
        "select(item_id,sid_0,sid_1,sid_2);sort(item_id);"
        "cast-int64;contiguous-c;sha256"
    ),
    "matrix_manifest_path": str(pathlib.Path(matrix_manifest_text).resolve()),
    "matrix_manifest_sha256": matrix_manifest_sha256,
    "verifier_output_path": str(verifier_output.resolve()),
}
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_text = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
)
temporary = pathlib.Path(temporary_text)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY

printf 'preverified anchor accepted: %s\n' "$ACCEPTANCE_PATH"

CANONICAL_RUNS=()
while IFS= read -r run_name; do
    CANONICAL_RUNS+=("$run_name")
done < <(
    "$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
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
export SID_CURRENT_LOG_PATH_FILE="$CURRENT_LOG_PATH"
export SID_PUBLIC_CURRENT_LOG_PATH_FILE="$PUBLIC_CURRENT_LOG_PATH"

set_status "MATRIX completed=0/30"
completed_count=0
for run_name in "${CANONICAL_RUNS[@]}"; do
    marker="$COMPLETION_DIR/$run_name.json"
    if [[ -f "$marker" ]]; then
        completed_count=$((completed_count + 1))
        set_status "MATRIX completed=$completed_count/30 skipped=$run_name"
        printf 'skip completed run=%s marker=%s\n' "$run_name" "$marker"
        continue
    fi

    set_current_run "$run_name"
    set_status "MATRIX completed=$completed_count/30 running=$run_name"
    printf 'starting run=%s gpu=%s\n' "$run_name" "$GPU_ID"
    if "$POINT_RUNNER" "$run_name" "$GPU_ID"; then
        :
    else
        point_status=$?
        if (( point_status == 4 )) && [[ -f "$marker" ]]; then
            printf 'run completed concurrently: %s\n' "$run_name"
        else
            exit "$point_status"
        fi
    fi

    completed_count=$((completed_count + 1))
    "$PYTHON_BIN" "$EXPORT_SCRIPT" \
        --reports-root "$REPORTS_ROOT" \
        --output "$PARTIAL_EXPORT_PATH" \
        --matrix-manifest "$MANIFEST_PATH" \
        --completion-dir "$COMPLETION_DIR"
    set_status "MATRIX completed=$completed_count/30"
done

set_current_run "final-export"
set_status "EXPORT_FINAL"
"$PYTHON_BIN" "$EXPORT_SCRIPT" \
    --reports-root "$REPORTS_ROOT" \
    --output "$FINAL_EXPORT_PATH" \
    --matrix-manifest "$MANIFEST_PATH" \
    --completion-dir "$COMPLETION_DIR"

"$PYTHON_BIN" - "$FINAL_EXPORT_PATH" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 30, f"expected 30 final rows, received {len(rows)}"
print("final export ok: 30 rows")
PY

set_current_run "none"
set_status "FINISHED completed=30/30"
RUNNER_FINISHED=1
printf 'quality matrix resume finished successfully\n'
printf 'final results: %s\n' "$FINAL_EXPORT_PATH"
