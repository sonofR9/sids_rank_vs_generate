from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


def path_average_similarity(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str = "normalized_embed",
) -> float | None:
    """Average within-path cosine similarity, averaged equally over collisions."""

    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    required = {"item_id", embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )

    collisions = (
        sids.select("item_id", *sid_cols)
        .join(embeddings.select("item_id", embedding_col), on="item_id", how="inner")
        .group_by(sid_cols)
        .agg(
            pl.col(embedding_col).alias("embeddings"),
            pl.len().alias("collision_size"),
        )
        .filter(pl.col("collision_size") > 1)
    )

    similarities: list[float] = []
    for vectors in collisions["embeddings"]:
        matrix = np.asarray(vectors.to_list(), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )
        pairwise = normalized @ normalized.T
        count = matrix.shape[0]
        similarities.append(
            float((pairwise.sum() - np.trace(pairwise)) / (count * (count - 1)))
        )

    return float(np.mean(similarities)) if similarities else None


def codebook_quality(
    sids: pl.DataFrame,
    codebook_sizes: Sequence[int],
) -> dict[str, float]:
    """Calculate load balance, prefix coverage, and utilization per SID level."""

    sid_cols = get_sid_cols(sids)
    sizes = tuple(int(size) for size in codebook_sizes)
    if len(sizes) != len(sid_cols):
        raise ValueError("one codebook size is required for every SID level")
    if any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    if sids.is_empty():
        raise ValueError("semantic ID frame must not be empty")

    metrics: dict[str, float] = {}
    load_balance: dict[str, list[float]] = {"p25": [], "p75": [], "p90": []}
    possible_paths = 1

    for level, (column, size) in enumerate(zip(sid_cols, sizes, strict=True), start=1):
        minimum = int(sids[column].min())
        maximum = int(sids[column].max())
        if minimum < 0 or maximum >= size:
            raise ValueError(
                f"{column} contains codes outside the configured range [0, {size})"
            )

        counts_by_code = sids.group_by(column).len(name="count")
        counts = np.zeros(size, dtype=np.float64)
        counts[counts_by_code[column].to_numpy()] = counts_by_code["count"].to_numpy()
        theoretical_mean = sids.height / size

        for percentile, label in ((25, "p25"), (75, "p75"), (90, "p90")):
            value = float(
                np.percentile(counts, percentile, method="inverted_cdf")
                / theoretical_mean
            )
            metrics[f"load_balance_{label}_at_{level}_layer"] = value
            load_balance[label].append(value)

        possible_paths *= size
        unique_prefixes = sids.select(sid_cols[:level]).unique().height
        metrics[f"codebook_coverage_at_{level}_layer"] = (
            unique_prefixes / possible_paths
        )
        metrics[f"codebook_utilization_rate_at_{level}_layer"] = (
            counts_by_code.height / size
        )

    for label, values in load_balance.items():
        metrics[f"load_balance_{label}_mean"] = float(np.mean(values))
    return metrics


def infer_codebook_sizes(
    sid_config: Any,
    sids: pl.DataFrame,
) -> tuple[int, ...]:
    """Read configured codebook sizes, falling back to observed zero-based ranges."""

    sid_cols = get_sid_cols(sids)
    configured = None
    if sid_config is not None:
        configured = getattr(sid_config, "codebook_sizes", None)
        if configured is None:
            configured = getattr(sid_config, "codebooks_sizes", None)
        if configured is None:
            size = getattr(sid_config, "codebook_size", None)
            levels = getattr(sid_config, "num_levels", None)
            if size is not None and levels is not None:
                configured = (int(size),) * int(levels)

    if configured is not None and len(configured) == len(sid_cols):
        return tuple(int(size) for size in configured)
    return tuple(int(sids[column].max()) + 1 for column in sid_cols)


# Compatibility aliases for the names introduced in PR #17.
PAS = path_average_similarity
codebook_load_balance_and_coverage_and_utitlization = codebook_quality
