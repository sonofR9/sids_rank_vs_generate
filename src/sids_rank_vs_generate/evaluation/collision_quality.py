from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


def _mean_pairwise_cosine(vectors: Sequence[Sequence[float]]) -> float:
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > 0,
    )
    # sum(normalized @ normalized.T) == squared norm of the vector sum.  This
    # avoids allocating an O(n^2) pairwise matrix for large per-code groups.
    vector_sum = normalized.sum(axis=0)
    diagonal = np.sum(normalized * normalized)
    count = matrix.shape[0]
    return float(
        (np.dot(vector_sum, vector_sum) - diagonal) / (count * (count - 1))
    )


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
        similarities.append(_mean_pairwise_cosine(vectors.to_list()))

    return float(np.mean(similarities)) if similarities else None


def intra_code_similarity_v2(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str = "normalized_embed",
) -> float | None:
    """Average cosine similarity among items assigned the same per-level code.

    Unlike :func:`path_average_similarity`, this metric does not group items by
    their complete semantic-ID path.  Every SID level is considered
    independently and each ``(level, code)`` group with at least two items
    contributes one equally weighted mean pairwise cosine similarity.
    """

    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    required = {"item_id", embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )

    items = sids.select("item_id", *sid_cols).join(
        embeddings.select("item_id", embedding_col),
        on="item_id",
        how="inner",
    )
    similarities: list[float] = []
    for sid_col in sid_cols:
        code_groups = (
            items.group_by(sid_col)
            .agg(
                pl.col(embedding_col).alias("embeddings"),
                pl.len().alias("code_size"),
            )
            .filter(pl.col("code_size") > 1)
        )
        for vectors in code_groups["embeddings"]:
            similarities.append(_mean_pairwise_cosine(vectors.to_list()))

    return float(np.mean(similarities)) if similarities else None


def prefix_coherence(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str = "normalized_embed",
) -> dict[str, float | None]:
    """Measure semantic coherence for every increasingly specific SID prefix.

    At depth ``d``, items are grouped by ``sid_0, ..., sid_{d-1}``.  Every
    non-singleton prefix contributes its mean pairwise cosine similarity and
    prefixes are weighted equally.
    """

    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    required = {"item_id", embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )

    items = sids.select("item_id", *sid_cols).join(
        embeddings.select("item_id", embedding_col),
        on="item_id",
        how="inner",
    )
    metrics: dict[str, float | None] = {}
    for level in range(1, len(sid_cols) + 1):
        groups = (
            items.group_by(sid_cols[:level])
            .agg(
                pl.col(embedding_col).alias("embeddings"),
                pl.len().alias("prefix_size"),
            )
            .filter(pl.col("prefix_size") > 1)
        )
        similarities = [
            _mean_pairwise_cosine(vectors.to_list())
            for vectors in groups["embeddings"]
        ]
        metrics[f"prefix_coherence_at_{level}"] = (
            float(np.mean(similarities)) if similarities else None
        )
    return metrics


def adjusted_semantic_variance_explained(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str = "normalized_embed",
) -> dict[str, float | None]:
    """Return chance-adjusted embedding variance explained by SID prefixes.

    The score compares within-prefix variance per residual degree of freedom
    with global variance per degree of freedom.  Unlike raw clustering
    R-squared, it does not become perfect merely because every item receives a
    unique prefix.  Unlike prefix coherence, it is item-weighted and measures
    separation relative to the catalog's global embedding variance.
    """

    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    required = {"item_id", embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )

    items = sids.select("item_id", *sid_cols).join(
        embeddings.select("item_id", embedding_col),
        on="item_id",
        how="inner",
    )
    matrix = np.asarray(items[embedding_col].to_list(), dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    total_sse = float(np.sum(centered * centered))
    metrics: dict[str, float | None] = {}
    for level in range(1, len(sid_cols) + 1):
        if total_sse == 0.0:
            metrics[f"adjusted_semantic_variance_explained_at_{level}"] = None
            continue
        groups = items.group_by(sid_cols[:level]).agg(
            pl.col(embedding_col).alias("embeddings")
        )
        if groups.height >= items.height:
            metrics[f"adjusted_semantic_variance_explained_at_{level}"] = None
            continue
        within_sse = 0.0
        for vectors in groups["embeddings"]:
            group = np.asarray(vectors.to_list(), dtype=np.float64)
            group_centered = group - group.mean(axis=0, keepdims=True)
            within_sse += float(np.sum(group_centered * group_centered))
        adjusted_explained = 1.0 - (
            (within_sse / (items.height - groups.height))
            / (total_sse / (items.height - 1))
        )
        metrics[f"adjusted_semantic_variance_explained_at_{level}"] = float(
            adjusted_explained
        )
    return metrics


def sid_distribution_quality(
    sids: pl.DataFrame,
    codebook_sizes: Sequence[int],
) -> dict[str, float | None]:
    """Measure collision exposure and how effectively each level refines paths."""

    sid_cols = get_sid_cols(sids)
    sizes = tuple(int(size) for size in codebook_sizes)
    if not sid_cols:
        raise ValueError("semantic ID frame has no sid_<level> columns")
    if len(sizes) != len(sid_cols):
        raise ValueError("one codebook size is required for every SID level")
    if any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    if sids.is_empty():
        raise ValueError("semantic ID frame must not be empty")
    for column, size in zip(sid_cols, sizes, strict=True):
        minimum = int(sids[column].min())
        maximum = int(sids[column].max())
        if minimum < 0 or maximum >= size:
            raise ValueError(
                f"{column} contains codes outside the configured range [0, {size})"
            )

    def entropy(counts: np.ndarray) -> float:
        probabilities = counts.astype(np.float64) / counts.sum()
        return float(-np.sum(probabilities * np.log(probabilities)))

    item_count = sids.height
    path_counts = sids.group_by(sid_cols).len()["len"].to_numpy()
    colliding_counts = path_counts[path_counts > 1]
    path_counts_float = path_counts.astype(np.float64)

    metrics: dict[str, float | None] = {
        "collision_item_rate": (
            float(colliding_counts.sum() / item_count)
            if len(colliding_counts)
            else 0.0
        ),
        "pair_collision_probability": (
            float(
                np.sum(path_counts_float * (path_counts_float - 1))
                / (item_count * (item_count - 1))
            )
            if item_count > 1
            else 0.0
        ),
    }

    previous_prefix_entropy = 0.0
    previous_pair_count = item_count * (item_count - 1) / 2
    for level, (column, size) in enumerate(zip(sid_cols, sizes, strict=True), start=1):
        prefix_counts = sids.group_by(sid_cols[:level]).len()["len"].to_numpy()
        current_prefix_entropy = entropy(prefix_counts)
        conditional_entropy = max(
            0.0,
            current_prefix_entropy - previous_prefix_entropy,
        )
        max_code_entropy = np.log(size)
        metrics[f"normalized_conditional_entropy_at_{level}_layer"] = (
            float(np.clip(conditional_entropy / max_code_entropy, 0.0, 1.0))
            if max_code_entropy > 0
            else 1.0
        )

        current_counts_float = prefix_counts.astype(np.float64)
        current_pair_count = float(
            np.sum(current_counts_float * (current_counts_float - 1)) / 2
        )
        metrics[f"pair_refinement_rate_at_{level}_layer"] = (
            float(1.0 - current_pair_count / previous_pair_count)
            if previous_pair_count > 0
            else None
        )

        if level == 1:
            active_branch_fraction = sids[column].n_unique() / size
        else:
            child_counts = sids.group_by(sid_cols[: level - 1]).agg(
                pl.col(column).n_unique().alias("active_children")
            )
            active_branch_fraction = (
                float(child_counts["active_children"].mean()) / size
            )
        metrics[f"mean_active_branch_fraction_at_{level}_layer"] = float(
            active_branch_fraction
        )

        previous_prefix_entropy = current_prefix_entropy
        previous_pair_count = current_pair_count

    return metrics


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
    utilization: list[float] = []
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
        utilization_rate = counts_by_code.height / size
        metrics[f"codebook_utilization_rate_at_{level}_layer"] = utilization_rate
        utilization.append(utilization_rate)

    for label, values in load_balance.items():
        metrics[f"load_balance_{label}_mean"] = float(np.mean(values))
    metrics["codebook_utilization_rate_total"] = float(np.mean(utilization))
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
