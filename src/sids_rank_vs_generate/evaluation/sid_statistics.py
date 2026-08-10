from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


def _occupancy_gini(counts: np.ndarray) -> float:
    """Return the Gini coefficient of positive full-SID occupancies."""

    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("occupancy counts must be non-empty and positive")
    if values.size == 1:
        return 0.0
    values.sort()
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    gini = float(
        (2.0 * np.dot(ranks, values)) / (values.size * values.sum())
        - (values.size + 1.0) / values.size
    )
    return float(np.clip(gini, 0.0, 1.0))


def _occupancy_entropy(counts: np.ndarray) -> tuple[float, float]:
    """Return Shannon entropy in nats and entropy normalized by log(K)."""

    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("occupancy counts must be non-empty and positive")
    probabilities = values / values.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / math.log(values.size) if values.size > 1 else 0.0
    return entropy, float(np.clip(normalized, 0.0, 1.0))


def calculate_sid_statistics(
    frame: pl.DataFrame,
    *,
    item_col: str = "item_id",
) -> dict[str, Any]:
    sid_columns = get_sid_cols(frame)
    if not sid_columns:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    if frame.is_empty():
        raise ValueError("semantic ID frame must not be empty")

    path_stats = frame.group_by(sid_columns).len(name="collision_size")
    collision_stats = path_stats.filter(pl.col("collision_size") > 1)
    item_count = frame.height
    unique_path_count = path_stats.height
    singleton_count = path_stats.filter(pl.col("collision_size") == 1).height

    percentiles = {}
    for percentile, label in (
        (0.9, "p90"),
        (0.95, "p95"),
        (0.99, "p99"),
        (0.999, "p999"),
    ):
        value = collision_stats["collision_size"].quantile(
            percentile, interpolation="nearest"
        )
        percentiles[label] = int(value) if value is not None else 0

    levels = {}
    for column in sid_columns:
        counts = frame.group_by(column).len(name="count")["count"]
        levels[column] = {
            "used_codes": frame[column].n_unique(),
            "min_code": int(frame[column].min()),
            "max_code": int(frame[column].max()),
            "min_items_per_code": int(counts.min()),
            "max_items_per_code": int(counts.max()),
            "mean_items_per_code": float(counts.mean()),
        }

    occupancies = path_stats["collision_size"].to_numpy()
    code_entropy, code_entropy_normalized = _occupancy_entropy(occupancies)

    return {
        "item_count": item_count,
        "levels": levels,
        "unique_paths": unique_path_count,
        "independent_code_rate": unique_path_count / item_count,
        "collision_rate": 1.0 - unique_path_count / item_count,
        "singleton_count": singleton_count,
        "singleton_rate": singleton_count / item_count,
        "max_collision": int(path_stats["collision_size"].max()),
        "collision_percentiles": percentiles,
        "code_gini": _occupancy_gini(occupancies),
        "code_entropy": code_entropy,
        "code_entropy_normalized": code_entropy_normalized,
    }


def tests_semantic_ids(
    df: pl.DataFrame,
    num_levels: int | None = None,
    codebook_size: int | None = None,
    item_col: str = "item_id",
) -> dict[str, Any]:
    """Backward-compatible name used by the original SID experiment branch."""

    del num_levels, codebook_size
    return calculate_sid_statistics(df, item_col=item_col)
