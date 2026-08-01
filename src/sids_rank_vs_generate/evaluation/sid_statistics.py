from __future__ import annotations

from typing import Any

import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


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
    for percentile, label in ((0.9, "p90"), (0.99, "p99"), (0.999, "p999")):
        value = collision_stats["collision_size"].quantile(
            percentile, interpolation="higher"
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
