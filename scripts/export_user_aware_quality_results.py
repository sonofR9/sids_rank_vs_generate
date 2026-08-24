#!/usr/bin/env python3
"""Экспорт завершенных matrix runs"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sids_rank_vs_generate.user_aware_quality_matrix import (
    EXPECTED_CATALOG_SIZE,
    GLOBAL_SID_ANCHOR_ID,
    GLOBAL_SID_CODEBOOK_SIZES,
    GLOBAL_SID_FAMILY,
    GLOBAL_SID_REFERENCE_SCOPE,
    LEGACY_POSTER_REFERENCE_MATCH,
    MATRIX_ID,
    PROTOCOL_ID,
    QUALITY_METRIC_SCHEMA,
    USER_AWARE_QUALITY_MATRIX,
    V1_FAMILY,
    V2_FAMILY,
)
from sids_rank_vs_generate.user_aware_quality_results import (
    COMPLETION_SCHEMA_VERSION,
    STATE_DIRECTORY_NAME,
    nested_value,
    read_json_object,
    resolve_report_file,
    sha256_file,
)


MANIFEST_SCHEMA_VERSION = 1
EXPECTED_QUALITY_SCOPE = "personalized_catalog"
EXPECTED_QUALITY_AGGREGATION = "per_user_catalog_macro"
EXPECTED_QUALITY_USER_COUNT = 256

OUTPUT_FIELDS = (
    "matrix_id",
    "run_name",
    "run_id",
    "result_dir",
    "protocol_id",
    "global_sid_anchor_id",
    "global_sid_reference_scope",
    "legacy_poster_reference_match",
    "catalog_size",
    "fit_fingerprint",
    "behavior_fingerprint",
    "x_scope",
    "x_aggregation",
    "quality_schema",
    "quality_user_count",
    "quality_user_fingerprint",
    "family",
    "series",
    "personal_codebook_sizes",
    "user_profile_weight",
    "fusion_weight",
    "level_permutation",
    "legacy_heldout_recall_at_100",
    "independent_code_rate",
    "collision_rate",
    "p95_collision_group_size",
    "g_max_macro",
    "g_max_across_users",
    "cur_total",
    "code_gini",
    "code_entropy",
    "intra_code_similarity",
    "prefix_coherence_1",
    "prefix_coherence_2",
    "prefix_coherence_3",
    "valid_sid_rate",
)


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_sequence(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array")
    return value


def _require_text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _require_number(value: Any, description: str) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be a finite number")
    return value


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported matrix manifest schema_version")
    if manifest.get("matrix_id") != MATRIX_ID:
        raise ValueError("matrix manifest has an unexpected matrix_id")
    if manifest.get("quality_metric_schema") != QUALITY_METRIC_SCHEMA:
        raise ValueError("matrix manifest has an unexpected quality schema")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("matrix manifest has an unexpected protocol_id")
    if protocol.get("expected_catalog_size") != EXPECTED_CATALOG_SIZE:
        raise ValueError("matrix manifest has an unexpected catalog size")
    fixed_global_sid = _require_mapping(
        manifest.get("fixed_global_sid"),
        "manifest fixed_global_sid",
    )
    expected_global_sid = {
        "family": GLOBAL_SID_FAMILY,
        "codebook_sizes": list(GLOBAL_SID_CODEBOOK_SIZES),
        "anchor_id": GLOBAL_SID_ANCHOR_ID,
        "reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
        "legacy_poster_reference_match": (
            LEGACY_POSTER_REFERENCE_MATCH
        ),
    }
    if fixed_global_sid != expected_global_sid:
        raise ValueError(
            "matrix manifest has unexpected global SID anchor provenance"
        )

    runs = _require_sequence(manifest.get("runs"), "manifest runs")
    if manifest.get("run_count") != len(runs):
        raise ValueError("matrix manifest run_count does not match runs")
    expected_run_names = {run.run_name for run in USER_AWARE_QUALITY_MATRIX}
    canonical_by_name = {
        run.run_name: run.to_dict() for run in USER_AWARE_QUALITY_MATRIX
    }
    if len(runs) != len(expected_run_names):
        raise ValueError(
            "matrix manifest must contain the complete 30-run matrix"
        )
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw_run in enumerate(runs):
        run = _require_mapping(raw_run, f"manifest run {index}")
        run_name = _require_text(run.get("run_name"), f"manifest run {index} name")
        if run_name in by_name:
            raise ValueError(f"duplicate run_name in matrix manifest: {run_name}")
        personal = _require_mapping(
            run.get("personal_sid"),
            f"manifest run {run_name} personal_sid",
        )
        codebook_sizes = _require_sequence(
            personal.get("codebook_sizes"),
            f"manifest run {run_name} codebook_sizes",
        )
        permutation = _require_sequence(
            personal.get("level_permutation"),
            f"manifest run {run_name} level_permutation",
        )
        if not codebook_sizes or len(codebook_sizes) != len(permutation):
            raise ValueError(f"manifest run {run_name} has invalid SID levels")
        tags = _require_mapping(run.get("tags"), f"manifest run {run_name} tags")
        expected_tags = {
            "matrix_id": MATRIX_ID,
            "matrix_run_id": run_name,
            "protocol_id": PROTOCOL_ID,
            "quality_metric_schema": QUALITY_METRIC_SCHEMA,
            "expected_catalog_size": str(EXPECTED_CATALOG_SIZE),
            "global_sid_anchor_id": GLOBAL_SID_ANCHOR_ID,
            "global_sid_reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
            "legacy_poster_reference_match": (
                LEGACY_POSTER_REFERENCE_MATCH
            ),
            "x_scope": "user_catalog_macro",
            "quality_aggregation": EXPECTED_QUALITY_AGGREGATION,
        }
        for key, expected in expected_tags.items():
            if tags.get(key) != expected:
                raise ValueError(
                    f"manifest run {run_name} has unexpected tag {key}"
                )
        if run != canonical_by_name.get(run_name):
            raise ValueError(
                f"manifest run {run_name} differs from the canonical matrix"
            )
        by_name[run_name] = run
    if set(by_name) != expected_run_names:
        raise ValueError("matrix manifest run names differ from the canonical matrix")
    return by_name


def _quality_columns(
    quality: dict[str, Any],
    *,
    level_count: int,
) -> dict[str, Any]:
    columns = {
        "independent_code_rate": quality.get("independent_code_rate"),
        "collision_rate": quality.get("collision_rate"),
        "p95_collision_group_size": nested_value(
            quality, "collision_percentiles", "p95"
        ),
        "g_max_macro": quality.get("max_collision"),
        "g_max_across_users": quality.get("max_collision_across_users"),
        "cur_total": nested_value(
            quality,
            "codebook_quality",
            "codebook_utilization_rate_total",
        ),
        "code_gini": quality.get("code_gini"),
        "code_entropy": quality.get("code_entropy"),
        "intra_code_similarity": quality.get("intra_code_similarity"),
        "prefix_coherence_1": quality.get("prefix_coherence_at_1"),
        "prefix_coherence_2": quality.get("prefix_coherence_at_2"),
        "prefix_coherence_3": quality.get("prefix_coherence_at_3"),
        "valid_sid_rate": quality.get("valid_sid_rate"),
    }
    required = (
        "independent_code_rate",
        "collision_rate",
        "p95_collision_group_size",
        "g_max_macro",
        "g_max_across_users",
        "cur_total",
        "code_gini",
        "code_entropy",
        "intra_code_similarity",
        "prefix_coherence_1",
        "valid_sid_rate",
    )
    if level_count >= 2:
        required = (*required, "prefix_coherence_2")
    if level_count >= 3:
        required = (*required, "prefix_coherence_3")
    for name in required:
        _require_number(columns.get(name), f"quality metric {name}")
    return columns


def _record_from_completion(
    marker_path: Path,
    reports_root: Path,
    manifest_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    marker = read_json_object(marker_path)
    if marker.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise ValueError(f"unsupported completion schema: {marker_path}")
    if marker.get("matrix_id") != MATRIX_ID:
        raise ValueError(f"completion has unexpected matrix_id: {marker_path}")
    if marker.get("status") != "FINISHED" or marker.get("exit_code") != 0:
        raise ValueError(f"completion is not successful: {marker_path}")
    run_name = _require_text(marker.get("run_name"), "completion run_name")
    matrix = manifest_by_name.get(run_name)
    if matrix is None:
        raise ValueError(f"completion run is outside matrix manifest: {run_name}")
    if marker_path.stem != run_name:
        raise ValueError("completion filename must match run_name")
    marker_run_id = _require_text(marker.get("run_id"), "completion run_id")
    metrics_path = resolve_report_file(
        reports_root,
        _require_text(marker.get("metrics_path"), "completion metrics_path"),
    )
    if sha256_file(metrics_path) != marker.get("metrics_sha256"):
        raise ValueError(f"metrics checksum mismatch for {run_name}")

    payload = read_json_object(metrics_path)
    tracking = _require_mapping(payload.get("tracking"), "metrics tracking")
    if tracking.get("enabled") is not True:
        raise ValueError(f"MLflow tracking was not enabled for {run_name}")
    if tracking.get("run_name") != run_name:
        raise ValueError(f"metrics run_name mismatch for {run_name}")
    if tracking.get("run_id") != marker_run_id:
        raise ValueError(f"metrics run_id mismatch for {run_name}")
    if nested_value(payload, "config", "protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol mismatch for {run_name}")
    configured_catalog_size = nested_value(
        payload,
        "config",
        "expected_catalog_size",
    )
    if configured_catalog_size != EXPECTED_CATALOG_SIZE:
        raise ValueError(f"expected catalog contract mismatch for {run_name}")
    if nested_value(payload, "data", "protocol_id") != PROTOCOL_ID:
        raise ValueError(f"data protocol mismatch for {run_name}")
    if nested_value(payload, "data", "catalog_size") != EXPECTED_CATALOG_SIZE:
        raise ValueError(f"catalog size mismatch for {run_name}")
    if nested_value(payload, "data", "validation_reuses_test") is not True:
        raise ValueError(f"legacy heldout contract mismatch for {run_name}")
    fit_fingerprint = _require_text(
        nested_value(payload, "data", "fit_fingerprint"),
        f"fit fingerprint for {run_name}",
    )
    raw_behavior_fingerprint = nested_value(
        payload,
        "user_sid",
        "fit_behavior",
        "fingerprint",
    )
    if matrix["family"] == V2_FAMILY:
        behavior_fingerprint: str | None = _require_text(
            raw_behavior_fingerprint,
            f"behavior fingerprint for {run_name}",
        )
    elif matrix["family"] == V1_FAMILY:
        if raw_behavior_fingerprint is not None and not (
            isinstance(raw_behavior_fingerprint, str)
            and not raw_behavior_fingerprint.strip()
        ):
            raise ValueError(
                f"v1 behavior fingerprint must be empty for {run_name}"
            )
        behavior_fingerprint = None
    else:
        raise ValueError(f"unsupported matrix family for {run_name}")

    quality = _require_mapping(
        nested_value(payload, "user_sid", "catalog_quality"),
        f"catalog quality for {run_name}",
    )
    if quality.get("schema_version") != QUALITY_METRIC_SCHEMA:
        raise ValueError(f"quality schema mismatch for {run_name}")
    if quality.get("scope") != EXPECTED_QUALITY_SCOPE:
        raise ValueError(f"quality scope mismatch for {run_name}")
    if quality.get("aggregation") != EXPECTED_QUALITY_AGGREGATION:
        raise ValueError(f"quality aggregation mismatch for {run_name}")
    if quality.get("user_count") != EXPECTED_QUALITY_USER_COUNT:
        raise ValueError(f"quality user count mismatch for {run_name}")
    if quality.get("catalog_items_per_user") != EXPECTED_CATALOG_SIZE:
        raise ValueError(f"quality catalog size mismatch for {run_name}")
    sample = _require_mapping(
        quality.get("user_sample"),
        f"quality user sample for {run_name}",
    )
    if sample.get("selected_user_count") != EXPECTED_QUALITY_USER_COUNT:
        raise ValueError(f"selected quality user count mismatch for {run_name}")
    user_fingerprint = _require_text(
        sample.get("user_sample_fingerprint"),
        f"quality user fingerprint for {run_name}",
    )

    recall = _require_number(
        nested_value(
            payload,
            "evaluations",
            "generation",
            "metrics",
            "recall@100",
        ),
        f"Recall@100 for {run_name}",
    )
    personal_sid = _require_mapping(
        matrix.get("personal_sid"),
        f"manifest personal SID for {run_name}",
    )
    codebook_sizes = _require_sequence(
        personal_sid.get("codebook_sizes"),
        f"manifest codebook sizes for {run_name}",
    )
    quality_columns = _quality_columns(quality, level_count=len(codebook_sizes))

    result_dir = _require_text(marker.get("result_dir"), "completion result_dir")
    if Path(result_dir) != Path(marker["metrics_path"]).parent:
        raise ValueError(f"completion result_dir mismatch for {run_name}")
    return {
        "matrix_id": MATRIX_ID,
        "run_name": run_name,
        "run_id": marker_run_id,
        "result_dir": result_dir,
        "protocol_id": PROTOCOL_ID,
        "global_sid_anchor_id": GLOBAL_SID_ANCHOR_ID,
        "global_sid_reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
        "legacy_poster_reference_match": (
            LEGACY_POSTER_REFERENCE_MATCH
        ),
        "catalog_size": EXPECTED_CATALOG_SIZE,
        "fit_fingerprint": fit_fingerprint,
        "behavior_fingerprint": behavior_fingerprint,
        "x_scope": quality["scope"],
        "x_aggregation": quality["aggregation"],
        "quality_schema": quality["schema_version"],
        "quality_user_count": quality["user_count"],
        "quality_user_fingerprint": user_fingerprint,
        "family": matrix["family"],
        "series": ",".join(matrix["series"]),
        "personal_codebook_sizes": "x".join(
            str(value) for value in codebook_sizes
        ),
        "user_profile_weight": personal_sid["user_profile_weight"],
        "fusion_weight": personal_sid["fusion_weight"],
        "level_permutation": "".join(
            str(value) for value in personal_sid["level_permutation"]
        ),
        "legacy_heldout_recall_at_100": recall,
        **quality_columns,
    }


def collect_completed_records(
    reports_root: Path,
    completion_dir: Path,
    manifest_by_name: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not completion_dir.is_dir():
        marker_paths: list[Path] = []
    else:
        marker_paths = sorted(completion_dir.glob("*.json"))
    records_by_name: dict[str, dict[str, Any]] = {}
    run_ids: set[str] = set()
    for marker_path in marker_paths:
        record = _record_from_completion(
            marker_path,
            reports_root,
            manifest_by_name,
        )
        run_name = str(record["run_name"])
        run_id = str(record["run_id"])
        if run_name in records_by_name:
            raise ValueError(f"duplicate completed run_name: {run_name}")
        if run_id in run_ids:
            raise ValueError(f"duplicate completed MLflow run_id: {run_id}")
        records_by_name[run_name] = record
        run_ids.add(run_id)

    fingerprints = {
        str(record["quality_user_fingerprint"])
        for record in records_by_name.values()
    }
    if len(fingerprints) > 1:
        raise ValueError("quality user fingerprint differs across completed runs")
    fit_fingerprints = {
        str(record["fit_fingerprint"])
        for record in records_by_name.values()
    }
    if len(fit_fingerprints) > 1:
        raise ValueError("fit fingerprint differs across completed runs")
    behavior_fingerprints = {
        str(record["behavior_fingerprint"])
        for record in records_by_name.values()
        if record["family"] == V2_FAMILY
    }
    if len(behavior_fingerprints) > 1:
        raise ValueError(
            "behavior fingerprint differs across completed v2 runs"
        )

    records = [records_by_name[name] for name in sorted(records_by_name)]
    missing = sorted(set(manifest_by_name) - set(records_by_name))
    return records, missing


def atomic_write_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument(
        "--completion-dir",
        type=Path,
        help="Defaults to REPORTS_ROOT/.user-aware-quality-state/completed",
    )
    args = parser.parse_args()

    manifest = read_json_object(args.matrix_manifest)
    manifest_by_name = validate_manifest(manifest)
    completion_dir = args.completion_dir or (
        args.reports_root / STATE_DIRECTORY_NAME / "completed"
    )
    records, missing = collect_completed_records(
        args.reports_root,
        completion_dir,
        manifest_by_name,
    )
    if not records:
        raise SystemExit("no successfully completed user-aware quality results found")
    atomic_write_csv(args.output, records)
    print(f"wrote {len(records)} rows to {args.output}")
    print(f"coverage: {len(records)}/{len(manifest_by_name)}")
    print(f"missing_count: {len(missing)}")
    if missing:
        print("missing_run_names:")
        for run_name in missing:
            print(f"  {run_name}")


if __name__ == "__main__":
    main()
