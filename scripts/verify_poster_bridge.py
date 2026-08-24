#!/usr/bin/env python3
"""Проверить bridge перед user-aware матрицей"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

POSTER_FULL_33148_PROTOCOL_ID = "poster-full-33148-v1"
POSTER_FULL_33148_DATA_CONTRACT = {
    "interactions": 634_642,
    "train_interactions": 630_095,
    "fit_interactions": 630_095,
    "validation_interactions": 4_547,
    "test_interactions": 4_547,
    "users": 8_032,
    "catalog_size": 33_148,
    "test_start_timestamp": 25_878_770.0,
    "validation_start_timestamp": 25_878_770.0,
}
POSTER_RQKMEANS_L3_C256_REFERENCE = {
    "independent_code_rate": 0.8627971521660432,
    "collision_rate": 0.13720284783395675,
    "collision_p95": 5.0,
    "max_collision": 28,
    "codebook_utilization_rate_total": 1.0,
    "code_gini": 0.12706173879215976,
    "code_entropy": 10.174084041926339,
    "intra_code_similarity": 0.9449189484497289,
    "prefix_coherence_at_1": 0.7103009254205972,
    "prefix_coherence_at_2": 0.8610243591478416,
    "prefix_coherence_at_3": 0.9449189484497289,
    "valid_sid_rate": 1.0,
    "legacy_heldout_recall_at_100": 0.09143621766280106,
}
POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE = {
    "independent_code_rate": 0.8640943646675516,
    "collision_rate": 0.13590563533244837,
    "collision_p95": 5.0,
    "max_collision": 22,
    "codebook_utilization_rate_total": 1.0,
    "code_gini": 0.12609535684607587,
    "code_entropy": 10.17628737954354,
    "intra_code_similarity": 0.9452018556661516,
    "prefix_coherence_at_1": 0.7072719681937971,
    "prefix_coherence_at_2": 0.8605393418109137,
    "prefix_coherence_at_3": 0.9452018556661516,
    "valid_sid_rate": 1.0,
    "legacy_heldout_recall_at_100": 0.08563782337198929,
}
LEGACY_HISTORICAL_PROFILE = "legacy-historical"
CURRENT_V100_FP32_PROFILE = "v100-fp32-20260812"
DEFAULT_REFERENCE_PROFILE = LEGACY_HISTORICAL_PROFILE
REFERENCE_PROFILES = {
    LEGACY_HISTORICAL_PROFILE: POSTER_RQKMEANS_L3_C256_REFERENCE,
    CURRENT_V100_FP32_PROFILE: POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE,
}
CURRENT_V100_FP32_FIT_FINGERPRINT = (
    "d049f69c6373eed26ef3f847a31a5dae665cd8c8e9789b7e5b1c6687bc7c6d8e"
)
POSTER_GENERATION_EXAMPLES = {
    "train": 594_041,
    "validation": 4_484,
    "test": 4_484,
}
POSTER_SID_RECIPE = {
    "class": "RQKMeansPipeline",
    "item_count": 33_148,
    "columns": ["sid_0", "sid_1", "sid_2"],
}
POSTER_SID_CONFIG = {
    "num_levels": 3,
    "codebook_size": 256,
    "codebook_sizes": None,
    "kmeans_batch_size": 16_384,
    "predict_batch_size": 65_536,
}
POSTER_GENERATION_RECIPE = {
    "max_history_len": 16,
    "min_history_len": 5,
    "d_model": 128,
    "d_ff": 512,
    "num_heads": 4,
    "num_layers": 4,
    "dropout": 0.1,
    "batch_size": 1_024,
    "batch_size_val": 64,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "max_epochs": 30,
    "grad_clip": 1.0,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4,
    "num_beams": 150,
    "recall_k": 100,
    "constrain_to_catalog": True,
    "resolve_collisions": False,
    "max_train_examples": None,
    "max_validation_examples": None,
    "max_test_examples": None,
    "use_user_profile": False,
    "initialize_item_embeddings_from_content": False,
    "sid_only_validation_during_training": False,
    "validation_metric": None,
    "num_levels": 3,
    "codebook_size": 256,
    "codebook_sizes": [256, 256, 256],
    "rq_num_levels": 3,
    "semantic_history_weight": 0.0,
    "auxiliary_item_loss_weight": 0.0,
    "logq_mode": "none",
    "use_bf16": False,
}

COUNT_DERIVED_QUALITY_KEYS = frozenset(
    {
        "independent_code_rate",
        "collision_rate",
        "collision_p95",
        "max_collision",
        "codebook_utilization_rate_total",
        "code_gini",
        "code_entropy",
        "valid_sid_rate",
    }
)
COSINE_QUALITY_KEYS = frozenset(
    {
        "intra_code_similarity",
        "prefix_coherence_at_1",
        "prefix_coherence_at_2",
        "prefix_coherence_at_3",
    }
)
COUNT_DERIVED_TOLERANCE = 1e-12
COSINE_TOLERANCE = 1e-6
RECALL_TOLERANCE = 0.003


def _nested(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("bridge result is missing " + ".".join(path))
        value = value[key]
    return value


def _matches_exactly(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            numeric = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric) and numeric == expected
    return actual == expected


def _validate_exact_mapping(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"bridge {label} is missing {key}")
        actual_value = actual[key]
        if not _matches_exactly(actual_value, expected_value):
            raise ValueError(
                f"bridge {label} mismatch for {key}: expected "
                f"{expected_value!r}, received {actual_value!r}"
            )


def _validate_data_contract(
    summary: dict[str, Any],
    reference_profile: str,
) -> None:
    _validate_exact_mapping(
        summary,
        POSTER_FULL_33148_DATA_CONTRACT,
        label="data",
    )
    if summary.get("protocol_id") != POSTER_FULL_33148_PROTOCOL_ID:
        raise ValueError("bridge data used a different protocol")
    if summary.get("validation_reuses_test") is not True:
        raise ValueError("bridge data did not reuse the legacy heldout")
    if (
        reference_profile == CURRENT_V100_FP32_PROFILE
        and summary.get("fit_fingerprint") != CURRENT_V100_FP32_FIT_FINGERPRINT
    ):
        raise ValueError(
            "bridge data fit_fingerprint does not match the current V100/FP32 "
            "reference"
        )


def _validate_recipe(payload: dict[str, Any]) -> None:
    config = _nested(payload, "config")
    _validate_exact_mapping(
        config,
        {
            "dataset_size": "50m",
            "core_min_interactions_per_item": 5,
            "test_days": 7.0,
            "timestamp_unit_seconds": 5.0,
            "catalog_from_fit": False,
            "latest_interactions": None,
            "enabled_stages": ["generation"],
            "expected_catalog_size": 33_148,
        },
        label="experiment recipe",
    )
    if "user_sid" not in payload:
        raise ValueError("bridge result is missing user_sid")
    if payload["user_sid"] is not None:
        raise ValueError("bridge recipe must be content-only")

    sid = _nested(payload, "sid")
    _validate_exact_mapping(sid, POSTER_SID_RECIPE, label="SID recipe")
    _validate_exact_mapping(
        _nested(sid, "config"),
        POSTER_SID_CONFIG,
        label="SID config",
    )

    generation = _nested(payload, "evaluations", "generation")
    _validate_exact_mapping(
        generation,
        {
            "target_sid_type": "global",
            "loaded_checkpoint": None,
        },
        label="generation recipe",
    )
    _validate_exact_mapping(
        _nested(generation, "examples"),
        POSTER_GENERATION_EXAMPLES,
        label="generation examples",
    )
    _validate_exact_mapping(
        _nested(generation, "config"),
        POSTER_GENERATION_RECIPE,
        label="generation config",
    )


def _metric_tolerance(key: str) -> float:
    if key in COUNT_DERIVED_QUALITY_KEYS:
        return COUNT_DERIVED_TOLERANCE
    if key in COSINE_QUALITY_KEYS:
        return COSINE_TOLERANCE
    if key == "legacy_heldout_recall_at_100":
        return RECALL_TOLERANCE
    raise KeyError(f"bridge metric has no declared tolerance: {key}")


def _validate_numeric_reference(
    key: str,
    actual: Any,
    expected: float,
) -> float:
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bridge metric {key} is not numeric: {actual!r}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"bridge metric {key} is not finite: {actual!r}")
    tolerance = _metric_tolerance(key)
    if abs(numeric - float(expected)) > tolerance:
        raise ValueError(
            f"bridge mismatch for {key}: expected {expected}, "
            f"received {actual}, tolerance {tolerance}"
        )
    return numeric


def verify(
    payload: dict[str, Any],
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> dict[str, float]:
    try:
        reference = REFERENCE_PROFILES[reference_profile]
    except KeyError as error:
        supported = ", ".join(sorted(REFERENCE_PROFILES))
        raise ValueError(
            f"unknown bridge reference profile {reference_profile!r}; "
            f"expected one of: {supported}"
        ) from error
    if _nested(payload, "config", "protocol_id") != POSTER_FULL_33148_PROTOCOL_ID:
        raise ValueError("bridge used a different data protocol")
    _validate_data_contract(_nested(payload, "data"), reference_profile)
    _validate_recipe(payload)

    statistics = _nested(payload, "sid", "statistics")
    observed = {
        "independent_code_rate": _nested(
            statistics, "independent_code_rate"
        ),
        "collision_rate": _nested(statistics, "collision_rate"),
        "collision_p95": _nested(
            statistics, "collision_percentiles", "p95"
        ),
        "max_collision": _nested(statistics, "max_collision"),
        "codebook_utilization_rate_total": _nested(
            statistics,
            "codebook_quality",
            "codebook_utilization_rate_total",
        ),
        "code_gini": _nested(statistics, "code_gini"),
        "code_entropy": _nested(statistics, "code_entropy"),
        "intra_code_similarity": _nested(
            statistics, "intra_code_similarity"
        ),
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
        "legacy_heldout_recall_at_100": _nested(
            payload,
            "evaluations",
            "generation",
            "metrics",
            "recall@100",
        ),
    }
    validated = {
        key: _validate_numeric_reference(key, observed[key], expected)
        for key, expected in reference.items()
    }
    validation_recall = _nested(
        payload,
        "evaluations",
        "generation",
        "validation_metrics",
        "recall@100",
    )
    try:
        validation_recall = float(validation_recall)
    except (TypeError, ValueError) as error:
        raise ValueError("bridge validation recall is not numeric") from error
    if not math.isfinite(validation_recall) or abs(
        validation_recall - validated["legacy_heldout_recall_at_100"]
    ) > COUNT_DERIVED_TOLERANCE:
        raise ValueError("bridge validation and test heldout recall differ")
    return validated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(REFERENCE_PROFILES),
        default=DEFAULT_REFERENCE_PROFILE,
        help="reference series used for the strict bridge comparison",
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    observed = verify(payload, reference_profile=args.profile)
    print(
        json.dumps(
            {
                "reference_profile": args.profile,
                "metrics": observed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
