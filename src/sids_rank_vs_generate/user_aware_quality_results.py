from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .user_aware_quality_matrix import MATRIX_ID


COMPLETION_SCHEMA_VERSION = 1
STATE_DIRECTORY_NAME = ".user-aware-quality-state"
RUN_DIRECTORY_NAME = ".user-aware-quality-runs"
_SAFE_RUN_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")


def nested_value(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _relative_file(path: Path, root: Path, *, description: str) -> str:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{description} must be inside reports root") from error
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved_path}")
    return relative.as_posix()


def resolve_report_file(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("completion metrics_path must be a relative path")
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("completion metrics_path escapes reports root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"completed metrics file is missing: {resolved}")
    return resolved


def validate_run_name(run_name: str) -> None:
    if not run_name or _SAFE_RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run_name contains unsafe path characters")


def create_completion_marker(
    *,
    reports_root: Path,
    attempt_reports_dir: Path,
    completion_dir: Path,
    run_name: str,
    attempt_id: str,
    gpu_id: str,
    log_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Создать marker завершенного результата"""

    validate_run_name(run_name)
    if not attempt_id:
        raise ValueError("attempt_id must not be empty")
    if not gpu_id.isdigit():
        raise ValueError("gpu_id must be a non-negative integer")
    metrics_paths = sorted(attempt_reports_dir.rglob("metrics.json"))
    if len(metrics_paths) != 1:
        raise ValueError(
            "successful attempt must contain exactly one metrics.json; "
            f"found {len(metrics_paths)}"
        )
    metrics_path = metrics_paths[0]
    payload = read_json_object(metrics_path)
    tracking = payload.get("tracking")
    if not isinstance(tracking, dict) or tracking.get("enabled") is not True:
        raise ValueError("successful matrix result must contain MLflow tracking")
    if tracking.get("run_name") != run_name:
        raise ValueError("metrics run_name does not match the matrix run")
    run_id = tracking.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("successful matrix result must contain an MLflow run_id")
    recall = nested_value(
        payload,
        "evaluations",
        "generation",
        "metrics",
        "recall@100",
    )
    if (
        not isinstance(recall, (int, float))
        or isinstance(recall, bool)
        or not math.isfinite(float(recall))
    ):
        raise ValueError("successful matrix result must contain finite recall@100")

    relative_metrics_path = _relative_file(
        metrics_path,
        reports_root,
        description="metrics.json",
    )
    relative_result_dir = str(Path(relative_metrics_path).parent)
    marker_path = completion_dir / f"{run_name}.json"
    if marker_path.exists():
        raise FileExistsError(f"completion marker already exists: {marker_path}")
    marker = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "status": "FINISHED",
        "exit_code": 0,
        "run_name": run_name,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "gpu_id": gpu_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metrics_path": relative_metrics_path,
        "metrics_sha256": sha256_file(metrics_path),
        "result_dir": relative_result_dir,
        "log_path": str(log_path.expanduser().resolve()),
    }
    atomic_write_json(marker_path, marker)
    return marker_path, marker
