from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_user_aware_quality_point.sh"
RUN_NAME = "uaq30-v1-anchor-l3-c256"


def _payload(run_id: str = "mlflow-launcher") -> dict:
    return {
        "data": {
            "catalog_size": 33_148,
            "protocol_id": "poster-full-33148-v1",
            "validation_reuses_test": True,
            "fit_fingerprint": "fit",
        },
        "user_sid": {
            "catalog_quality": {
                "schema_version": "sid-quality-v1",
                "scope": "personalized_catalog",
                "aggregation": "per_user_catalog_macro",
            }
        },
        "evaluations": {"generation": {"metrics": {"recall@100": 0.1}}},
        "config": {
            "protocol_id": "poster-full-33148-v1",
            "expected_catalog_size": 33_148,
        },
        "tracking": {
            "enabled": True,
            "run_name": RUN_NAME,
            "run_id": run_id,
        },
    }


def _fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_flock = bin_dir / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\nexit \"${FAKE_FLOCK_EXIT:-0}\"\n",
        encoding="utf-8",
    )
    fake_flock.chmod(0o755)

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")
    visible_gpu_path = tmp_path / "visible-gpu.txt"
    wrapper = bin_dir / "python-wrapper"
    wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
    if [[ "$argument" == "run.py" ]]; then
        mkdir -p "$SID_REPORTS_DIR/fake/result"
        cp "$FAKE_METRICS_PAYLOAD" "$SID_REPORTS_DIR/fake/result/metrics.json"
        printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$FAKE_VISIBLE_GPU_PATH"
        exit "${FAKE_RUN_EXIT:-0}"
    fi
done
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return bin_dir, wrapper, visible_gpu_path


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir, wrapper, visible_gpu_path = _fake_tools(tmp_path)
    reports = tmp_path / "reports"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "SID_REPO_ROOT": str(ROOT),
        "SID_PYTHON_BIN": str(wrapper),
        "SID_REPORTS_DIR": str(reports),
        "FAKE_METRICS_PAYLOAD": str(tmp_path / "payload.json"),
        "FAKE_VISIBLE_GPU_PATH": str(visible_gpu_path),
        "REAL_PYTHON": sys.executable,
    }
    return environment, reports, visible_gpu_path


def test_launcher_requires_explicit_gpu(tmp_path: Path) -> None:
    environment, _reports, _visible_gpu = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "GPU_ID is required" in result.stderr


def test_launcher_sets_gpu_and_creates_one_success_marker(tmp_path: Path) -> None:
    environment, reports, visible_gpu = _environment(tmp_path)

    first = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME, "3"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert visible_gpu.read_text(encoding="utf-8").strip() == "3"
    marker_path = (
        reports
        / ".user-aware-quality-state"
        / "completed"
        / f"{RUN_NAME}.json"
    )
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "FINISHED"
    assert marker["run_name"] == RUN_NAME
    assert marker["gpu_id"] == "3"

    second = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME, "3"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 4
    assert "already completed" in second.stderr


def test_launcher_refuses_busy_per_run_lock(tmp_path: Path) -> None:
    environment, reports, _visible_gpu = _environment(tmp_path)
    environment["FAKE_FLOCK_EXIT"] = "1"

    result = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME, "1"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert "already locked" in result.stderr
    marker_path = (
        reports
        / ".user-aware-quality-state"
        / "completed"
        / f"{RUN_NAME}.json"
    )
    assert not marker_path.exists()


def test_failed_training_does_not_create_success_marker(tmp_path: Path) -> None:
    environment, reports, _visible_gpu = _environment(tmp_path)
    environment["FAKE_RUN_EXIT"] = "17"

    result = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME, "1"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 17
    marker_path = (
        reports
        / ".user-aware-quality-state"
        / "completed"
        / f"{RUN_NAME}.json"
    )
    assert not marker_path.exists()


def test_global_sid_anchor_failure_does_not_create_success_marker(
    tmp_path: Path,
) -> None:
    environment, reports, _visible_gpu = _environment(tmp_path)
    environment["SID_GLOBAL_SID_CANONICAL_SHA256"] = "0" * 64

    result = subprocess.run(
        ["bash", str(LAUNCHER), RUN_NAME, "1"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    marker_path = (
        reports
        / ".user-aware-quality-state"
        / "completed"
        / f"{RUN_NAME}.json"
    )
    assert not marker_path.exists()
