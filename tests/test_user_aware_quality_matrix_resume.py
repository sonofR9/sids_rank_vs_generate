from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_user_aware_quality_matrix_resume.sh"
PROFILE = "v100-fp32-20260812"
CANONICAL_SHA256 = "3" * 64
PRIORITY_RUNS = [
    "uaq30-v1-anchor-l3-c256",
    "uaq30-v2-fusion-weight-0",
    "uaq30-v2-anchor-l3-c256",
    "uaq30-v1-capacity-l4-c256",
    "uaq30-v2-capacity-l4-c256",
    "uaq30-v1-capacity-l3-c1024",
    "uaq30-v2-capacity-l3-c1024",
]
RUN_NAMES = PRIORITY_RUNS + [f"fake-run-{index:02d}" for index in range(7, 30)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "flock",
        "#!/usr/bin/env bash\nexit \"${FAKE_FLOCK_EXIT:-0}\"\n",
    )

    python_wrapper = _executable(
        bin_dir / "python-wrapper",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" && "${2:-}" == *"np.ascontiguousarray"* ]]; then
    printf '%s\n' "$FAKE_CANONICAL_SHA256"
    exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
    )

    verify_log = tmp_path / "verify.args"
    verify_script = _executable(
        tmp_path / "verify.py",
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["FAKE_VERIFY_LOG"]).write_text(
    "\\n".join(sys.argv[1:]) + "\\n", encoding="utf-8"
)
profile = sys.argv[sys.argv.index("--profile") + 1]
print(json.dumps({"reference_profile": profile, "metrics": {}}))
""",
    )

    plan_script = _executable(
        tmp_path / "plan.py",
        """#!/usr/bin/env python3
import json
import os

names = json.loads(os.environ["FAKE_RUN_NAMES"])
print(json.dumps({
    "matrix_id": "user-aware-quality-30-current-anchor-v1",
    "run_count": len(names),
    "protocol": {
        "expected_catalog_size": 33148,
        "id": "poster-full-33148-v1",
    },
    "fixed_global_sid": {
        "anchor_id": "current-rqkmeans-l3-c256-seed42-v1",
        "reference_scope": "current_anchor",
    },
    "runs": [{"run_name": name} for name in names],
}))
""",
    )

    export_log = tmp_path / "export.args.jsonl"
    export_script = _executable(
        tmp_path / "export.py",
        """#!/usr/bin/env python3
import csv
import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
with pathlib.Path(os.environ["FAKE_EXPORT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
output = pathlib.Path(arguments[arguments.index("--output") + 1])
completion = pathlib.Path(arguments[arguments.index("--completion-dir") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
markers = sorted(completion.glob("*.json"))
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["run_name"])
    writer.writeheader()
    for marker in markers:
        writer.writerow({"run_name": marker.stem})
""",
    )

    point_log = tmp_path / "point.log"
    point_runner = _executable(
        tmp_path / "point.sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|%s\n' "$1" "$2" "$PYTHONPATH" >> "$FAKE_POINT_LOG"
mkdir -p "$SID_MATRIX_STATE_DIR/completed"
printf '{}\n' > "$SID_MATRIX_STATE_DIR/completed/$1.json"
""",
    )

    metrics_path = tmp_path / "bridge" / "metrics.json"
    semantic_ids_path = tmp_path / "bridge" / "semantic_ids.parquet"
    metrics_path.parent.mkdir()
    metrics_path.write_text('{"preverified": true}\n', encoding="utf-8")
    semantic_ids_path.write_bytes(b"preverified semantic ids\n")

    reports_root = tmp_path / "local-reports"
    matrix_state = tmp_path / "shared-state"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "REAL_PYTHON": sys.executable,
        "FAKE_CANONICAL_SHA256": CANONICAL_SHA256,
        "FAKE_VERIFY_LOG": str(verify_log),
        "FAKE_EXPORT_LOG": str(export_log),
        "FAKE_POINT_LOG": str(point_log),
        "FAKE_RUN_NAMES": json.dumps(RUN_NAMES),
        "SID_REPO_ROOT": str(ROOT),
        "SID_PYTHON_BIN": str(python_wrapper),
        "SID_REPORTS_DIR": str(reports_root),
        "SID_MLFLOW_URI": "http://mlflow.invalid:5000",
        "SID_MATRIX_STATE_DIR": str(matrix_state),
        "SID_PREVERIFIED_BRIDGE_METRICS_PATH": str(metrics_path),
        "SID_PREVERIFIED_BRIDGE_METRICS_SHA256": _sha256(metrics_path),
        "SID_PREVERIFIED_BRIDGE_SIDS_PATH": str(semantic_ids_path),
        "SID_PREVERIFIED_BRIDGE_SIDS_SHA256": _sha256(semantic_ids_path),
        "SID_GLOBAL_SID_CANONICAL_SHA256": CANONICAL_SHA256,
        "SID_SOURCE_SHA256": "1" * 64,
        "SID_BRIDGE_VERIFY_SCRIPT": str(verify_script),
        "SID_MATRIX_PLAN_SCRIPT": str(plan_script),
        "SID_MATRIX_EXPORT_SCRIPT": str(export_script),
        "SID_MATRIX_POINT_RUNNER": str(point_runner),
    }
    paths = {
        "metrics": metrics_path,
        "semantic_ids": semantic_ids_path,
        "reports": reports_root,
        "state": matrix_state,
        "verify_log": verify_log,
        "export_log": export_log,
        "point_log": point_log,
    }
    return environment, paths


def test_resume_accepts_anchor_and_runs_all_points_sequentially(
    tmp_path: Path,
) -> None:
    environment, paths = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "2"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    gpu_state = paths["state"] / "resume-current-anchor" / "gpu-2"
    queue_log = (gpu_state / "queue.log").read_text(encoding="utf-8")
    assert result.returncode == 0, queue_log + result.stdout + result.stderr
    point_rows = paths["point_log"].read_text(encoding="utf-8").splitlines()
    assert [row.split("|", 1)[0] for row in point_rows] == RUN_NAMES
    assert all(row.split("|")[1] == "2" for row in point_rows)
    assert all(row.split("|")[2] == str(ROOT / "src") for row in point_rows)

    verify_arguments = paths["verify_log"].read_text(encoding="utf-8").splitlines()
    assert verify_arguments == [str(paths["metrics"]), "--profile", PROFILE]

    acceptance = json.loads(
        (gpu_state / "preverified-anchor.acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["status"] == "ACCEPTED"
    assert acceptance["verifier_profile"] == PROFILE
    assert acceptance["bridge_metrics_sha256"] == _sha256(paths["metrics"])
    assert acceptance["bridge_semantic_ids_sha256"] == _sha256(
        paths["semantic_ids"]
    )
    assert acceptance["global_sid_canonical_sha256"] == CANONICAL_SHA256

    export_calls = [
        json.loads(line)
        for line in paths["export_log"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(export_calls) == 31
    completion_dir = str(paths["state"] / "completed")
    assert all(
        call[call.index("--completion-dir") + 1] == completion_dir
        for call in export_calls
    )
    partial = paths["reports"] / (
        "user-aware-quality-results.current-anchor.partial.csv"
    )
    final = paths["reports"] / (
        "user-aware-quality-results.current-anchor.final.csv"
    )
    assert partial.is_file()
    assert final.is_file()
    assert partial != final
    assert (gpu_state / "status").read_text(encoding="utf-8").strip() == (
        "FINISHED completed=30/30"
    )


def test_resume_rejects_anchor_hash_before_starting_a_point(
    tmp_path: Path,
) -> None:
    environment, paths = _environment(tmp_path)
    environment["SID_PREVERIFIED_BRIDGE_SIDS_SHA256"] = "0" * 64

    result = subprocess.run(
        ["bash", str(RUNNER), "1"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 12
    gpu_state = paths["state"] / "resume-current-anchor" / "gpu-1"
    queue_log = (gpu_state / "queue.log").read_text(encoding="utf-8")
    assert "semantic IDs SHA256 mismatch" in queue_log
    assert not paths["point_log"].exists()
    status = (
        gpu_state / "status"
    ).read_text(encoding="utf-8")
    assert status.startswith("FAILED exit=12 current=preverified-anchor")
