#!/usr/bin/env python3
"""Проверить неизменность глобальных SID в matrix point"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_poster_bridge import (  # noqa: E402
    CURRENT_V100_FP32_PROFILE,
    POSTER_FULL_33148_PROTOCOL_ID,
    POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE,
    POSTER_SID_RECIPE,
    _nested,
    _validate_data_contract,
    _validate_exact_mapping,
    _validate_numeric_reference,
)


CANONICAL_HASH_FORMAT = "sorted-row-major-little-endian-int64-v1"
CANONICAL_COLUMNS = ("item_id", "sid_0", "sid_1", "sid_2")
EXPECTED_CATALOG_SIZE = 33_148
EXPECTED_CODEBOOK_SIZES = (256, 256, 256)
EXPECTED_KMEANS_BATCH_SIZE = 16_384
EXPECTED_PREDICT_BATCH_SIZE = 65_536
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_INTEGER_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
)
_QUALITY_KEYS = tuple(
    key
    for key in POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE
    if key != "legacy_heldout_recall_at_100"
)


def _normalize_expected_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("expected canonical SID SHA256 must be 64 hexadecimal chars")
    return normalized


def _resolve_semantic_ids_path(
    payload: dict[str, Any],
    metrics_path: Path,
) -> Path:
    raw_path = _nested(payload, "artifacts", "semantic_ids")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("metrics artifacts.semantic_ids must be a non-empty path")
    artifact_path = Path(raw_path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path.cwd() / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"semantic_ids artifact does not exist: {artifact_path} "
            f"(declared by {metrics_path})"
        )
    return artifact_path


def canonical_semantic_ids_sha256(path: Path) -> str:
    """Хешировать item_id и SID после сортировки и Int64 cast"""

    frame = pl.read_parquet(path)
    if len(frame.columns) != len(CANONICAL_COLUMNS) or set(frame.columns) != set(
        CANONICAL_COLUMNS
    ):
        raise ValueError(
            "semantic_ids artifact must contain exactly: "
            + ", ".join(CANONICAL_COLUMNS)
        )
    if frame.height != EXPECTED_CATALOG_SIZE:
        raise ValueError(
            "semantic_ids artifact catalog mismatch: expected "
            f"{EXPECTED_CATALOG_SIZE}, received {frame.height}"
        )
    if any(frame[column].null_count() for column in CANONICAL_COLUMNS):
        raise ValueError("semantic_ids artifact contains null canonical values")
    for column in CANONICAL_COLUMNS:
        if frame.schema[column] not in _INTEGER_DTYPES:
            raise ValueError(
                f"semantic_ids column {column} must have an integer dtype"
            )
    if frame["item_id"].n_unique() != EXPECTED_CATALOG_SIZE:
        raise ValueError("semantic_ids artifact contains duplicate item_id values")

    canonical = frame.select(CANONICAL_COLUMNS).sort("item_id").with_columns(
        pl.col(column).cast(pl.Int64, strict=True)
        for column in CANONICAL_COLUMNS
    )
    for column in CANONICAL_COLUMNS[1:]:
        minimum = canonical[column].min()
        maximum = canonical[column].max()
        if minimum is None or maximum is None or minimum < 0 or maximum >= 256:
            raise ValueError(
                f"semantic_ids column {column} contains values outside [0, 255]"
            )

    matrix = np.asarray(canonical.to_numpy(), dtype="<i8", order="C")
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _observed_quality(payload: dict[str, Any]) -> dict[str, Any]:
    statistics = _nested(payload, "sid", "statistics")
    return {
        "independent_code_rate": _nested(statistics, "independent_code_rate"),
        "collision_rate": _nested(statistics, "collision_rate"),
        "collision_p95": _nested(statistics, "collision_percentiles", "p95"),
        "max_collision": _nested(statistics, "max_collision"),
        "codebook_utilization_rate_total": _nested(
            statistics,
            "codebook_quality",
            "codebook_utilization_rate_total",
        ),
        "code_gini": _nested(statistics, "code_gini"),
        "code_entropy": _nested(statistics, "code_entropy"),
        "intra_code_similarity": _nested(statistics, "intra_code_similarity"),
        "prefix_coherence_at_1": _nested(
            statistics, "prefix_coherence_at_1"
        ),
        "prefix_coherence_at_2": _nested(
            statistics, "prefix_coherence_at_2"
        ),
        "prefix_coherence_at_3": _nested(
            statistics, "prefix_coherence_at_3"
        ),
        "valid_sid_rate": _nested(statistics, "valid_sid_rate"),
    }


def _validate_global_sid_config(config: dict[str, Any]) -> None:
    _validate_exact_mapping(
        config,
        {
            "kmeans_batch_size": EXPECTED_KMEANS_BATCH_SIZE,
            "predict_batch_size": EXPECTED_PREDICT_BATCH_SIZE,
        },
        label="global SID config",
    )
    configured_sizes = config.get("codebook_sizes")
    if configured_sizes is None:
        num_levels = config.get("num_levels")
        codebook_size = config.get("codebook_size")
        if (
            not isinstance(num_levels, int)
            or isinstance(num_levels, bool)
            or not isinstance(codebook_size, int)
            or isinstance(codebook_size, bool)
        ):
            raise ValueError(
                "global SID config must declare integer num_levels and "
                "codebook_size"
            )
        effective_sizes = (codebook_size,) * num_levels
    else:
        if not isinstance(configured_sizes, (list, tuple)) or any(
            not isinstance(size, int) or isinstance(size, bool)
            for size in configured_sizes
        ):
            raise ValueError("global SID codebook_sizes must contain integers")
        effective_sizes = tuple(configured_sizes)
    if effective_sizes != EXPECTED_CODEBOOK_SIZES:
        raise ValueError(
            "global SID effective codebook sizes mismatch: expected "
            f"{EXPECTED_CODEBOOK_SIZES}, received {effective_sizes}"
        )


def verify(
    payload: dict[str, Any],
    *,
    metrics_path: Path,
    expected_sha256: str,
    profile: str = CURRENT_V100_FP32_PROFILE,
) -> dict[str, Any]:
    """Проверить протокол, global SID quality и канонический artifact"""

    if profile != CURRENT_V100_FP32_PROFILE:
        raise ValueError(
            "global SID anchor requires the current reference profile "
            f"{CURRENT_V100_FP32_PROFILE!r}"
        )
    if _nested(payload, "config", "protocol_id") != POSTER_FULL_33148_PROTOCOL_ID:
        raise ValueError("global SID anchor used a different data protocol")
    _validate_data_contract(_nested(payload, "data"), profile)

    sid = _nested(payload, "sid")
    _validate_exact_mapping(sid, POSTER_SID_RECIPE, label="global SID recipe")
    _validate_global_sid_config(_nested(sid, "config"))

    observed = _observed_quality(payload)
    validated_quality = {
        key: _validate_numeric_reference(
            key,
            observed[key],
            POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE[key],
        )
        for key in _QUALITY_KEYS
    }

    expected = _normalize_expected_sha256(expected_sha256)
    semantic_ids_path = _resolve_semantic_ids_path(payload, metrics_path)
    actual = canonical_semantic_ids_sha256(semantic_ids_path)
    if actual != expected:
        raise ValueError(
            "global semantic_ids canonical SHA256 mismatch: "
            f"expected {expected}, received {actual}"
        )
    return {
        "reference_profile": profile,
        "canonical_hash_format": CANONICAL_HASH_FORMAT,
        "canonical_columns": list(CANONICAL_COLUMNS),
        "semantic_ids": str(semantic_ids_path),
        "canonical_sha256": actual,
        "quality_metrics": validated_quality,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=(CURRENT_V100_FP32_PROFILE,),
        required=True,
    )
    args = parser.parse_args(argv)
    metrics_path = args.metrics_json.expanduser().resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    result = verify(
        payload,
        metrics_path=metrics_path,
        expected_sha256=args.expected_sha256,
        profile=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
