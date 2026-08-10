from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


def _macro_group_cosine_similarity(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    embedding_col: str,
) -> float | None:
    """Average within-group pairwise cosine, with equal weight per group."""

    groups = (
        sids.select("item_id", *group_cols)
        .join(embeddings.select("item_id", embedding_col), on="item_id", how="inner")
        .group_by(list(group_cols))
        .agg(
            pl.col(embedding_col).alias("embeddings"),
            pl.len().alias("group_size"),
        )
        .filter(pl.col("group_size") > 1)
    )

    similarities: list[float] = []
    for vectors in groups["embeddings"]:
        matrix = np.asarray(vectors.to_list(), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )
        vector_sum = normalized.sum(axis=0)
        diagonal = float(np.square(normalized).sum())
        count = matrix.shape[0]
        similarity = (
            np.dot(vector_sum, vector_sum) - diagonal
        ) / (count * (count - 1))
        similarities.append(
            float(np.clip(similarity, -1.0, 1.0))
        )

    return float(np.mean(similarities)) if similarities else None


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

    return _macro_group_cosine_similarity(
        sids,
        embeddings,
        group_cols=sid_cols,
        embedding_col=embedding_col,
    )


def prefix_coherence(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str = "normalized_embed",
) -> dict[str, float | None]:
    """Return macro-average semantic coherence for every SID prefix depth."""

    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    required = {"item_id", embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )

    return {
        f"prefix_coherence_at_{level}": _macro_group_cosine_similarity(
            sids,
            embeddings,
            group_cols=sid_cols[:level],
            embedding_col=embedding_col,
        )
        for level in range(1, len(sid_cols) + 1)
    }


def valid_sid_rate(
    sids: pl.DataFrame,
    codebook_sizes: Sequence[int],
) -> float:
    """Fraction of rows with a complete, integral, in-range semantic path."""

    sid_cols = get_sid_cols(sids)
    sizes = tuple(int(size) for size in codebook_sizes)
    if len(sizes) != len(sid_cols):
        raise ValueError("one codebook size is required for every SID level")
    if any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    if sids.is_empty():
        raise ValueError("semantic ID frame must not be empty")
    if any(not sids.schema[column].is_integer() for column in sid_cols):
        return 0.0

    valid = pl.lit(True)
    for column, size in zip(sid_cols, sizes, strict=True):
        valid = (
            valid
            & pl.col(column).is_not_null()
            & (pl.col(column) >= 0)
            & (pl.col(column) < size)
        )
    return float(sids.select(valid.mean().alias("valid_sid_rate")).item())


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
    total_used_codes = 0
    total_available_codes = 0

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
        total_used_codes += counts_by_code.height
        total_available_codes += size

    for label, values in load_balance.items():
        metrics[f"load_balance_{label}_mean"] = float(np.mean(values))
    metrics["codebook_utilization_rate_total"] = (
        total_used_codes / total_available_codes
    )
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
