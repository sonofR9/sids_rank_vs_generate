from __future__ import annotations

import csv
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sids_rank_vs_generate.user_aware_quality_matrix import (
    EXPECTED_CATALOG_SIZE,
    GLOBAL_SID_ANCHOR_ID,
    GLOBAL_SID_REFERENCE_SCOPE,
    LEGACY_POSTER_REFERENCE_MATCH,
    MATRIX_ID,
    PROTOCOL_ID,
    QUALITY_METRIC_SCHEMA,
    USER_AWARE_QUALITY_MATRIX,
    V2_FAMILY,
    matrix_manifest,
)
from sids_rank_vs_generate.user_aware_quality_results import (
    COMPLETION_SCHEMA_VERSION,
    create_completion_marker,
    read_json_object,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_user_aware_quality_results.py"
)
SPEC = importlib.util.spec_from_file_location("quality_export", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _quality(levels: int = 3, *, fingerprint: str = "users-fingerprint"):
    quality = {
        "schema_version": QUALITY_METRIC_SCHEMA,
        "scope": "personalized_catalog",
        "aggregation": "per_user_catalog_macro",
        "user_count": 256,
        "catalog_items_per_user": EXPECTED_CATALOG_SIZE,
        "valid_sid_rate": 1.0,
        "independent_code_rate": 0.8,
        "collision_rate": 0.2,
        "collision_percentiles": {"p95": 4.0},
        "max_collision": 5.0,
        "max_collision_across_users": 9.0,
        "codebook_quality": {"codebook_utilization_rate_total": 0.7},
        "code_gini": 0.1,
        "code_entropy": 8.0,
        "intra_code_similarity": 0.6,
        "user_sample": {
            "selected_user_count": 256,
            "user_sample_fingerprint": fingerprint,
        },
    }
    for level in range(1, levels + 1):
        quality[f"prefix_coherence_at_{level}"] = 0.5 + level / 100
    return quality


def _metrics_payload(run_name: str, run_id: str, *, levels: int = 3):
    family = next(
        run.family
        for run in USER_AWARE_QUALITY_MATRIX
        if run.run_name == run_name
    )
    fit_behavior = (
        {"fingerprint": "behavior-fingerprint"}
        if family == V2_FAMILY
        else None
    )
    return {
        "data": {
            "catalog_size": EXPECTED_CATALOG_SIZE,
            "protocol_id": PROTOCOL_ID,
            "validation_reuses_test": True,
            "fit_fingerprint": "fit-fingerprint",
        },
        "user_sid": {
            "catalog_quality": _quality(levels),
            "fit_behavior": fit_behavior,
        },
        "evaluations": {
            "generation": {"metrics": {"recall@100": 0.123}}
        },
        "config": {
            "protocol_id": PROTOCOL_ID,
            "expected_catalog_size": EXPECTED_CATALOG_SIZE,
        },
        "tracking": {
            "enabled": True,
            "experiment_name": "quality",
            "run_name": run_name,
            "run_id": run_id,
        },
    }


def _write_attempt(
    root: Path,
    run_name: str,
    run_id: str,
    *,
    levels: int = 3,
    attempt_id: str = "attempt-1",
):
    attempt = root / ".user-aware-quality-runs" / run_name / attempt_id
    result = attempt / "50m" / "RQKMeansPipeline" / "result"
    result.mkdir(parents=True)
    metrics_path = result / "metrics.json"
    metrics_path.write_text(
        json.dumps(_metrics_payload(run_name, run_id, levels=levels)),
        encoding="utf-8",
    )
    completion_dir = root / ".user-aware-quality-state" / "completed"
    marker_path, marker = create_completion_marker(
        reports_root=root,
        attempt_reports_dir=attempt,
        completion_dir=completion_dir,
        run_name=run_name,
        attempt_id=attempt_id,
        gpu_id="1",
        log_path=root / "run.log",
    )
    return metrics_path, marker_path, marker


def _two_run_manifest():
    return matrix_manifest()


def _rewrite_completed_metrics(
    metrics_path: Path,
    marker_path: Path,
    marker: dict,
    payload: dict,
) -> None:
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    from sids_rank_vs_generate.user_aware_quality_results import sha256_file

    marker["metrics_sha256"] = sha256_file(metrics_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_completion_marker_is_atomic_and_binds_metrics(tmp_path: Path) -> None:
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v1-anchor-l3-c256",
        "mlflow-1",
    )

    assert marker_path.is_file()
    assert marker["schema_version"] == COMPLETION_SCHEMA_VERSION
    assert marker["matrix_id"] == MATRIX_ID
    assert marker["status"] == "FINISHED"
    assert marker["exit_code"] == 0
    assert marker["run_id"] == "mlflow-1"
    assert marker["metrics_path"] == str(metrics_path.relative_to(tmp_path))
    assert not list(marker_path.parent.glob("*.tmp"))


def test_completion_rejects_duplicate_marker(tmp_path: Path) -> None:
    _write_attempt(tmp_path, "uaq30-v1-anchor-l3-c256", "mlflow-1")

    with pytest.raises(FileExistsError, match="already exists"):
        create_completion_marker(
            reports_root=tmp_path,
            attempt_reports_dir=(
                tmp_path
                / ".user-aware-quality-runs"
                / "uaq30-v1-anchor-l3-c256"
                / "attempt-1"
            ),
            completion_dir=(
                tmp_path / ".user-aware-quality-state" / "completed"
            ),
            run_name="uaq30-v1-anchor-l3-c256",
            attempt_id="attempt-1",
            gpu_id="1",
            log_path=tmp_path / "run.log",
        )


def test_manifest_validation_rejects_duplicate_names() -> None:
    manifest = _two_run_manifest()
    manifest["runs"][-1]["run_name"] = manifest["runs"][0]["run_name"]
    manifest["runs"][-1]["tags"]["matrix_run_id"] = manifest["runs"][0][
        "run_name"
    ]

    with pytest.raises(ValueError, match="duplicate run_name"):
        EXPORT.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 99, "schema_version"),
        ("matrix_id", "wrong", "matrix_id"),
        ("quality_metric_schema", "wrong", "quality schema"),
    ),
)
def test_manifest_validation_rejects_wrong_schema(
    field: str,
    value,
    message: str,
) -> None:
    manifest = _two_run_manifest()
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        EXPORT.validate_manifest(manifest)


def test_manifest_validation_rejects_modified_matrix_row() -> None:
    manifest = _two_run_manifest()
    manifest["runs"][0]["personal_sid"]["user_profile_weight"] = 999

    with pytest.raises(ValueError, match="canonical matrix"):
        EXPORT.validate_manifest(manifest)


def test_manifest_validation_rejects_legacy_anchor_claim() -> None:
    manifest = _two_run_manifest()
    manifest["fixed_global_sid"]["reference_scope"] = "legacy_reference"

    with pytest.raises(ValueError, match="anchor provenance"):
        EXPORT.validate_manifest(manifest)


def test_collect_is_one_row_per_completed_run_and_reports_missing(
    tmp_path: Path,
) -> None:
    manifest = _two_run_manifest()
    by_name = EXPORT.validate_manifest(manifest)
    _write_attempt(tmp_path, "uaq30-v1-anchor-l3-c256", "mlflow-1")

    records, missing = EXPORT.collect_completed_records(
        tmp_path,
        tmp_path / ".user-aware-quality-state" / "completed",
        by_name,
    )

    assert len(records) == 1
    assert records[0]["run_name"] == "uaq30-v1-anchor-l3-c256"
    assert records[0]["legacy_heldout_recall_at_100"] == 0.123
    assert records[0]["behavior_fingerprint"] is None
    assert records[0]["global_sid_anchor_id"] == GLOBAL_SID_ANCHOR_ID
    assert (
        records[0]["global_sid_reference_scope"]
        == GLOBAL_SID_REFERENCE_SCOPE
    )
    assert (
        records[0]["legacy_poster_reference_match"]
        == LEGACY_POSTER_REFERENCE_MATCH
    )
    assert len(missing) == 29
    assert "uaq30-v2-anchor-l3-c256" in missing


def test_collect_exports_required_v2_behavior_fingerprint(
    tmp_path: Path,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    _write_attempt(tmp_path, "uaq30-v2-anchor-l3-c256", "mlflow-1")

    records, _missing = EXPORT.collect_completed_records(
        tmp_path,
        tmp_path / ".user-aware-quality-state" / "completed",
        by_name,
    )

    assert records[0]["behavior_fingerprint"] == "behavior-fingerprint"


def test_collect_rejects_missing_v2_behavior_fingerprint(
    tmp_path: Path,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v2-anchor-l3-c256",
        "mlflow-1",
    )
    payload = read_json_object(metrics_path)
    payload["user_sid"]["fit_behavior"] = None
    _rewrite_completed_metrics(metrics_path, marker_path, marker, payload)

    with pytest.raises(ValueError, match="behavior fingerprint"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_nonempty_v1_behavior_fingerprint(
    tmp_path: Path,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v1-anchor-l3-c256",
        "mlflow-1",
    )
    payload = read_json_object(metrics_path)
    payload["user_sid"]["fit_behavior"] = {"fingerprint": "unexpected"}
    _rewrite_completed_metrics(metrics_path, marker_path, marker, payload)

    with pytest.raises(ValueError, match="v1 behavior fingerprint"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_tampered_metrics(tmp_path: Path) -> None:
    manifest = _two_run_manifest()
    by_name = EXPORT.validate_manifest(manifest)
    metrics_path, _marker_path, _marker = _write_attempt(
        tmp_path,
        "uaq30-v1-anchor-l3-c256",
        "mlflow-1",
    )
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_wrong_contract_and_missing_metric(tmp_path: Path) -> None:
    manifest = _two_run_manifest()
    by_name = EXPORT.validate_manifest(manifest)
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v1-anchor-l3-c256",
        "mlflow-1",
    )
    payload = read_json_object(metrics_path)
    del payload["user_sid"]["catalog_quality"]["code_gini"]
    _rewrite_completed_metrics(metrics_path, marker_path, marker, payload)

    with pytest.raises(ValueError, match="code_gini"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("config", "protocol_id"), "wrong", "protocol mismatch"),
        (("data", "catalog_size"), 1, "catalog size mismatch"),
        (
            ("user_sid", "catalog_quality", "schema_version"),
            "wrong",
            "quality schema mismatch",
        ),
        (
            ("user_sid", "catalog_quality", "scope"),
            "wrong",
            "quality scope mismatch",
        ),
        (
            ("user_sid", "catalog_quality", "aggregation"),
            "wrong",
            "quality aggregation mismatch",
        ),
        (
            ("evaluations", "generation", "metrics", "recall@100"),
            None,
            "Recall@100",
        ),
    ),
)
def test_collect_rejects_incompatible_result_contract(
    tmp_path: Path,
    path: tuple[str, ...],
    value,
    message: str,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v1-anchor-l3-c256",
        "mlflow-1",
    )
    payload = read_json_object(metrics_path)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _rewrite_completed_metrics(metrics_path, marker_path, marker, payload)

    with pytest.raises(ValueError, match=message):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_fingerprint_drift(tmp_path: Path) -> None:
    manifest = _two_run_manifest()
    by_name = EXPORT.validate_manifest(manifest)
    _write_attempt(tmp_path, "uaq30-v1-anchor-l3-c256", "mlflow-1")
    second_metrics, second_marker, marker = _write_attempt(
        tmp_path,
        "uaq30-v2-anchor-l3-c256",
        "mlflow-2",
    )
    payload = read_json_object(second_metrics)
    payload["user_sid"]["catalog_quality"]["user_sample"][
        "user_sample_fingerprint"
    ] = "different-users"
    _rewrite_completed_metrics(second_metrics, second_marker, marker, payload)

    with pytest.raises(ValueError, match="fingerprint differs"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_v2_behavior_fingerprint_drift(
    tmp_path: Path,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    _write_attempt(tmp_path, "uaq30-v2-anchor-l3-c256", "mlflow-1")
    metrics_path, marker_path, marker = _write_attempt(
        tmp_path,
        "uaq30-v2-capacity-l2-c256",
        "mlflow-2",
        levels=2,
    )
    payload = read_json_object(metrics_path)
    payload["user_sid"]["fit_behavior"]["fingerprint"] = "different"
    _rewrite_completed_metrics(metrics_path, marker_path, marker, payload)

    with pytest.raises(ValueError, match="behavior fingerprint differs"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_collect_rejects_duplicate_mlflow_run_id(tmp_path: Path) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    _write_attempt(tmp_path, "uaq30-v1-anchor-l3-c256", "same-run-id")
    _write_attempt(tmp_path, "uaq30-v2-anchor-l3-c256", "same-run-id")

    with pytest.raises(ValueError, match="duplicate completed MLflow run_id"):
        EXPORT.collect_completed_records(
            tmp_path,
            tmp_path / ".user-aware-quality-state" / "completed",
            by_name,
        )


def test_raw_metrics_without_completion_marker_are_not_exported(
    tmp_path: Path,
) -> None:
    by_name = EXPORT.validate_manifest(_two_run_manifest())
    raw = tmp_path / "raw" / "metrics.json"
    raw.parent.mkdir()
    run_name = "uaq30-v1-anchor-l3-c256"
    raw.write_text(
        json.dumps(_metrics_payload(run_name, "x")),
        encoding="utf-8",
    )
    assert run_name in by_name

    records, missing = EXPORT.collect_completed_records(
        tmp_path,
        tmp_path / ".user-aware-quality-state" / "completed",
        by_name,
    )

    assert records == []
    assert len(missing) == 30


def test_atomic_csv_has_stable_schema(tmp_path: Path) -> None:
    record = {field: field for field in EXPORT.OUTPUT_FIELDS}
    output = tmp_path / "results.csv"

    EXPORT.atomic_write_csv(output, [record])

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == EXPORT.OUTPUT_FIELDS
    assert not list(tmp_path.glob("*.tmp"))


def test_completion_rejects_failed_or_partial_payload(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    result = attempt / "result"
    result.mkdir(parents=True)
    (result / "metrics.json").write_text(
        json.dumps(_metrics_payload("uaq30-v1-anchor-l3-c256", "mlflow-1")),
        encoding="utf-8",
    )
    second = attempt / "other" / "metrics.json"
    second.parent.mkdir()
    second.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one metrics.json"):
        create_completion_marker(
            reports_root=tmp_path,
            attempt_reports_dir=attempt,
            completion_dir=tmp_path / "completed",
            run_name="uaq30-v1-anchor-l3-c256",
            attempt_id="attempt",
            gpu_id="1",
            log_path=tmp_path / "run.log",
        )
