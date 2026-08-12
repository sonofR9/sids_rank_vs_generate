from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols
from sids_rank_vs_generate.sids.user_aware import (
    user_sid_columns,
    validate_user_sid_output,
)


SID_QUALITY_SCHEMA_VERSION = "sid-quality-v1"


@dataclass(frozen=True)
class UserCatalogQualityConfig:
    """Настройки сопоставимой X-метрики персональных SID"""

    enabled: bool = False
    max_users: int = 256
    selection: str = "stable_hash"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_users <= 0:
            raise ValueError("max_users must be positive")
        if self.selection != "stable_hash":
            raise ValueError("selection must be 'stable_hash'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _canonical_identifier(value: Any) -> str:
    return json.dumps(
        {
            "type": type(value).__name__,
            "value": value,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def select_stable_quality_users(
    users: pl.DataFrame,
    config: UserCatalogQualityConfig,
    *,
    uid_col: str = "uid",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Выбирает пользователей независимо от порядка строк"""

    if uid_col not in users.columns:
        raise ValueError(f"users must contain {uid_col!r}")
    eligible = users.select(uid_col).drop_nulls().unique()
    if eligible.is_empty():
        raise ValueError("eligible users must not be empty")

    ranked = []
    for uid in eligible[uid_col].to_list():
        canonical = _canonical_identifier(uid)
        rank = hashlib.sha256(
            f"{config.seed}:{canonical}".encode("utf-8")
        ).digest()
        ranked.append((rank, canonical, uid))
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected_rows = ranked[: config.max_users]
    selected_values = [row[2] for row in selected_rows]
    selected = pl.DataFrame(
        {uid_col: pl.Series(selected_values, dtype=eligible.schema[uid_col])}
    )

    fingerprint = hashlib.sha256()
    for _rank, canonical, _uid in selected_rows:
        encoded = canonical.encode("utf-8")
        fingerprint.update(len(encoded).to_bytes(8, byteorder="big"))
        fingerprint.update(encoded)
    summary = {
        "selection": config.selection,
        "selection_source": "fit_users",
        "seed": config.seed,
        "eligible_user_count": eligible.height,
        "selected_user_count": selected.height,
        "user_sample_fingerprint": fingerprint.hexdigest(),
    }
    return selected, summary


def _validate_codebook_sizes(
    columns: Sequence[str],
    codebook_sizes: Sequence[int],
) -> tuple[int, ...]:
    sizes = tuple(int(size) for size in codebook_sizes)
    if len(sizes) != len(columns):
        raise ValueError("one codebook size is required for every SID level")
    if any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    return sizes


def _gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64, copy=False))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    positions = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(
        (2 * np.dot(positions, values) / (len(values) * values.sum()))
        - (len(values) + 1) / len(values)
    )


def _valid_sid_rows(
    frame: pl.DataFrame,
    columns: Sequence[str],
    sizes: Sequence[int],
) -> pl.DataFrame:
    valid = pl.lit(True)
    for column, size in zip(columns, sizes, strict=True):
        valid &= pl.col(column).is_between(0, size - 1, closed="both")
    return frame.filter(valid.fill_null(False))


def _validate_embeddings(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    item_col: str,
    embedding_col: str,
) -> None:
    required = {item_col, embedding_col}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(
            "embedding frame is missing columns: " + ", ".join(sorted(missing))
        )
    if embeddings[item_col].null_count():
        raise ValueError("embedding frame contains null item identifiers")
    if embeddings[item_col].n_unique() != embeddings.height:
        raise ValueError("embedding frame must contain one row per item")
    missing_items = (
        sids.select(item_col)
        .unique()
        .join(embeddings.select(item_col), on=item_col, how="anti")
    )
    if not missing_items.is_empty():
        raise ValueError("embedding frame does not cover the evaluated catalog")


def _mean_within_group_cosine(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    prefix: Sequence[str],
    *,
    item_col: str,
    embedding_col: str,
) -> float | None:
    grouped = (
        sids.select(item_col, *prefix)
        .join(
            embeddings.select(item_col, embedding_col),
            on=item_col,
            how="inner",
        )
        .group_by(list(prefix))
        .agg(
            pl.col(embedding_col).alias("vectors"),
            pl.len().alias("count"),
        )
        .filter(pl.col("count") > 1)
    )

    similarities: list[float] = []
    for vectors, count in grouped.select("vectors", "count").iter_rows():
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )
        total = normalized.sum(axis=0)
        similarities.append(
            float((np.dot(total, total) - count) / (count * (count - 1)))
        )
    return float(np.mean(similarities)) if similarities else None


def _normalized_embedding_matrix(
    catalog_items: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    item_col: str,
    embedding_col: str,
) -> np.ndarray:
    """Выравнивает эмбеддинги по каталогу и нормализует один раз"""

    _validate_embeddings(
        catalog_items,
        embeddings,
        item_col=item_col,
        embedding_col=embedding_col,
    )
    aligned = (
        catalog_items.select(item_col)
        .with_row_index("_catalog_row")
        .join(
            embeddings.select(item_col, embedding_col),
            on=item_col,
            how="left",
            validate="1:1",
        )
        .sort("_catalog_row")
    )
    matrix = np.asarray(aligned[embedding_col].to_list(), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(
            "embedding frame must contain non-empty vector embeddings"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > 0,
    )


def _mean_within_group_cosine_numpy(
    codes: np.ndarray,
    normalized_embeddings: np.ndarray,
    prefix_levels: int,
) -> float | None:
    """Считает средний cosine по группам без Polars join"""

    _, inverse, counts = np.unique(
        codes[:, :prefix_levels],
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if not np.any(counts > 1):
        return None

    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1,
            np.asarray([len(order)], dtype=np.int64),
        )
    )
    similarities: list[float] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        count = int(stop - start)
        if count <= 1:
            continue
        total = normalized_embeddings[order[start:stop]].sum(
            axis=0,
            dtype=np.float64,
        )
        similarities.append(
            float((np.dot(total, total) - count) / (count * (count - 1)))
        )
    return float(np.mean(similarities)) if similarities else None


def _calculate_sid_quality_numpy(
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
    normalized_embeddings: np.ndarray | None,
) -> dict[str, Any]:
    """NumPy-эквивалент calculate_sid_quality для полного каталога"""

    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] == 0:
        raise ValueError("semantic ID codes must be a non-empty matrix")
    sizes = tuple(int(size) for size in codebook_sizes)
    if len(sizes) != codes.shape[1]:
        raise ValueError("one codebook size is required for every SID level")
    if any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    if normalized_embeddings is not None:
        if normalized_embeddings.ndim != 2:
            raise ValueError("embedding matrix must be two-dimensional")
        if normalized_embeddings.shape[0] != codes.shape[0]:
            raise ValueError("embedding matrix must align with semantic ID rows")

    valid_mask = np.ones(codes.shape[0], dtype=bool)
    for level, size in enumerate(sizes):
        valid_mask &= (codes[:, level] >= 0) & (codes[:, level] < size)
    valid_codes = codes[valid_mask]
    if valid_codes.shape[0] == 0:
        raise ValueError("semantic ID frame has no valid semantic IDs")

    _, counts = np.unique(valid_codes, axis=0, return_counts=True)
    collisions = counts[counts > 1]
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_categories = min(
        valid_codes.shape[0],
        int(np.prod(sizes, dtype=object)),
    )
    normalized_entropy = (
        entropy / float(np.log(max_categories)) if max_categories > 1 else 0.0
    )
    utilization = [
        np.unique(valid_codes[:, level]).size / size
        for level, size in enumerate(sizes)
    ]
    codebook_quality = {
        f"codebook_utilization_rate_at_{level}_layer": float(value)
        for level, value in enumerate(utilization, start=1)
    }
    codebook_quality["codebook_utilization_rate_total"] = float(
        np.mean(utilization)
    )

    valid_count = valid_codes.shape[0]
    unique_paths = counts.size
    result: dict[str, Any] = {
        "schema_version": SID_QUALITY_SCHEMA_VERSION,
        "scope": "assigned_catalog",
        "valid_sid_definition": "assigned_codes_in_configured_range",
        "row_count": codes.shape[0],
        "valid_row_count": valid_count,
        "valid_sid_rate": valid_count / codes.shape[0],
        "unique_paths": unique_paths,
        "independent_code_rate": unique_paths / valid_count,
        "collision_rate": 1.0 - unique_paths / valid_count,
        "collision_percentiles": {
            "p95": (
                float(np.quantile(collisions, 0.95, method="nearest"))
                if len(collisions)
                else 0.0
            )
        },
        "max_collision": int(counts.max()),
        "codebook_quality": codebook_quality,
        "code_gini": _gini(counts),
        "code_entropy": entropy,
        "code_entropy_normalized": normalized_entropy,
    }

    if normalized_embeddings is not None:
        valid_embeddings = (
            normalized_embeddings
            if bool(valid_mask.all())
            else normalized_embeddings[valid_mask]
        )
        for level in range(1, valid_codes.shape[1] + 1):
            coherence = _mean_within_group_cosine_numpy(
                valid_codes,
                valid_embeddings,
                level,
            )
            if coherence is not None:
                result[f"prefix_coherence_at_{level}"] = coherence
        final_key = f"prefix_coherence_at_{valid_codes.shape[1]}"
        if final_key in result:
            result["intra_code_similarity"] = result[final_key]
    return result


def calculate_sid_quality(
    frame: pl.DataFrame,
    codebook_sizes: Sequence[int],
    embeddings: pl.DataFrame | None = None,
    *,
    sid_columns: Sequence[str] | None = None,
    item_col: str = "item_id",
    embedding_col: str | None = None,
) -> dict[str, Any]:
    """Считает метрики SID в схеме общих графиков"""

    if frame.is_empty():
        raise ValueError("semantic ID frame must not be empty")
    if item_col not in frame.columns:
        raise ValueError(f"semantic ID frame must contain {item_col!r}")
    columns = list(sid_columns) if sid_columns is not None else get_sid_cols(frame)
    if not columns:
        raise ValueError("semantic ID frame has no SID columns")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            "semantic ID frame is missing columns: " + ", ".join(sorted(missing))
        )
    for column in columns:
        if not frame.schema[column].is_integer():
            raise TypeError(f"semantic ID column {column} must be integer")

    sizes = _validate_codebook_sizes(columns, codebook_sizes)
    valid_sids = _valid_sid_rows(frame, columns, sizes)
    if valid_sids.is_empty():
        raise ValueError("semantic ID frame has no valid semantic IDs")

    paths = valid_sids.group_by(columns).len(name="count")
    counts = paths["count"].to_numpy()
    collisions = counts[counts > 1]
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_categories = min(valid_sids.height, int(np.prod(sizes, dtype=object)))
    normalized_entropy = (
        entropy / float(np.log(max_categories)) if max_categories > 1 else 0.0
    )

    utilization = [
        valid_sids[column].n_unique() / size
        for column, size in zip(columns, sizes, strict=True)
    ]
    codebook_quality = {
        f"codebook_utilization_rate_at_{level}_layer": float(value)
        for level, value in enumerate(utilization, start=1)
    }
    codebook_quality["codebook_utilization_rate_total"] = float(
        np.mean(utilization)
    )

    unique_paths = paths.height
    valid_count = valid_sids.height
    result: dict[str, Any] = {
        "schema_version": SID_QUALITY_SCHEMA_VERSION,
        "scope": "assigned_catalog",
        "valid_sid_definition": "assigned_codes_in_configured_range",
        "row_count": frame.height,
        "valid_row_count": valid_count,
        "valid_sid_rate": valid_count / frame.height,
        "unique_paths": unique_paths,
        "independent_code_rate": unique_paths / valid_count,
        "collision_rate": 1.0 - unique_paths / valid_count,
        "collision_percentiles": {
            "p95": (
                float(np.quantile(collisions, 0.95, method="nearest"))
                if len(collisions)
                else 0.0
            )
        },
        "max_collision": int(counts.max()),
        "codebook_quality": codebook_quality,
        "code_gini": _gini(counts),
        "code_entropy": entropy,
        "code_entropy_normalized": normalized_entropy,
    }

    if embeddings is not None:
        resolved_embedding_col = embedding_col or (
            "normalized_embed"
            if "normalized_embed" in embeddings.columns
            else "embed"
        )
        _validate_embeddings(
            valid_sids,
            embeddings,
            item_col=item_col,
            embedding_col=resolved_embedding_col,
        )
        for level in range(1, len(columns) + 1):
            coherence = _mean_within_group_cosine(
                valid_sids,
                embeddings,
                columns[:level],
                item_col=item_col,
                embedding_col=resolved_embedding_col,
            )
            if coherence is not None:
                result[f"prefix_coherence_at_{level}"] = coherence
        final_key = f"prefix_coherence_at_{len(columns)}"
        if final_key in result:
            result["intra_code_similarity"] = result[final_key]
    return result


def _metric_paths(levels: int) -> tuple[tuple[str, ...], ...]:
    scalar_paths = (
        ("valid_sid_rate",),
        ("independent_code_rate",),
        ("collision_rate",),
        ("max_collision",),
        ("code_gini",),
        ("code_entropy",),
        ("code_entropy_normalized",),
        ("collision_percentiles", "p95"),
        ("codebook_quality", "codebook_utilization_rate_total"),
        ("intra_code_similarity",),
    )
    utilization = tuple(
        ("codebook_quality", f"codebook_utilization_rate_at_{level}_layer")
        for level in range(1, levels + 1)
    )
    coherence = tuple(
        (f"prefix_coherence_at_{level}",) for level in range(1, levels + 1)
    )
    return scalar_paths + utilization + coherence


def _nested_value(values: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = values
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _set_nested_value(
    values: dict[str, Any],
    path: Sequence[str],
    value: float,
) -> None:
    target = values
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _macro_average_user_quality(
    per_user: Sequence[Mapping[str, Any]],
    *,
    catalog_size: int,
) -> dict[str, Any]:
    if not per_user:
        raise ValueError("at least one user is required for macro aggregation")
    levels = sum(
        key.startswith("codebook_utilization_rate_at_")
        for key in per_user[0]["codebook_quality"]
    )
    if levels == 0:
        raise ValueError("per-user quality has no codebook utilization levels")

    result: dict[str, Any] = {
        "schema_version": SID_QUALITY_SCHEMA_VERSION,
        "scope": "personalized_catalog",
        "aggregation": "per_user_catalog_macro",
        "valid_sid_definition": "assigned_codes_in_configured_range",
        "user_count": len(per_user),
        "catalog_items_per_user": catalog_size,
    }
    coverage: dict[str, int] = {}
    for path in _metric_paths(levels):
        observed = [
            float(value)
            for metrics in per_user
            if (value := _nested_value(metrics, path)) is not None
        ]
        if not observed:
            continue
        _set_nested_value(result, path, float(np.mean(observed)))
        coverage[".".join(path)] = len(observed)
    result["metric_user_counts"] = coverage
    result["max_collision_across_users"] = max(
        int(metrics["max_collision"]) for metrics in per_user
    )
    return result


def calculate_user_catalog_sid_quality_from_frame(
    frame: pl.DataFrame,
    codebook_sizes: Sequence[int],
    embeddings: pl.DataFrame | None = None,
    *,
    uid_col: str = "uid",
    item_col: str = "item_id",
    embedding_col: str | None = None,
) -> dict[str, Any]:
    """Считает macro-метрики по полному каталогу каждого пользователя"""

    if frame.is_empty():
        raise ValueError("personalized SID frame must not be empty")
    required = {uid_col, item_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "personalized SID frame is missing columns: "
            + ", ".join(sorted(missing))
        )
    if frame.select(uid_col, item_col).unique().height != frame.height:
        raise ValueError("personalized SID frame contains duplicate user-item pairs")
    columns = user_sid_columns(frame)
    if not columns:
        raise ValueError("personalized SID frame has no usid_<level> columns")

    user_count = frame[uid_col].n_unique()
    catalog_size = frame[item_col].n_unique()
    expected_rows = user_count * catalog_size
    if frame.height != expected_rows:
        raise ValueError("personalized quality requires a full user-catalog grid")
    item_coverage = frame.group_by(item_col).agg(pl.col(uid_col).n_unique().alias("n"))
    if item_coverage["n"].min() != user_count:
        raise ValueError("every user must be evaluated on the same catalog")

    per_user = [
        calculate_sid_quality(
            group,
            codebook_sizes,
            embeddings,
            sid_columns=columns,
            item_col=item_col,
            embedding_col=embedding_col,
        )
        for group in frame.partition_by(uid_col, maintain_order=True)
    ]
    return _macro_average_user_quality(per_user, catalog_size=catalog_size)


def calculate_user_catalog_sid_quality(
    user_sid_pipeline: Any,
    users: pl.DataFrame,
    catalog_items: pl.DataFrame,
    codebook_sizes: Sequence[int],
    embeddings: pl.DataFrame | None = None,
    *,
    uid_col: str = "uid",
    item_col: str = "item_id",
    embedding_col: str | None = None,
) -> dict[str, Any]:
    """Считает full-catalog macro без хранения полного декартова произведения"""

    if users.is_empty():
        raise ValueError("users must not be empty")
    if catalog_items.is_empty():
        raise ValueError("catalog items must not be empty")
    if uid_col not in users.columns:
        raise ValueError(f"users must contain {uid_col!r}")
    if item_col not in catalog_items.columns:
        raise ValueError(f"catalog items must contain {item_col!r}")
    if users[uid_col].null_count() or users[uid_col].n_unique() != users.height:
        raise ValueError("users must contain unique non-null identifiers")
    if (
        catalog_items[item_col].null_count()
        or catalog_items[item_col].n_unique() != catalog_items.height
    ):
        raise ValueError("catalog items must contain unique non-null identifiers")

    sizes = tuple(int(size) for size in codebook_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("codebook sizes must be positive")
    resolved_embedding_col: str | None = None
    normalized_embeddings: np.ndarray | None = None
    if embeddings is not None:
        resolved_embedding_col = embedding_col or (
            "normalized_embed"
            if "normalized_embed" in embeddings.columns
            else "embed"
        )
        normalized_embeddings = _normalized_embedding_matrix(
            catalog_items,
            embeddings,
            item_col=item_col,
            embedding_col=resolved_embedding_col,
        )

    per_user: list[dict[str, Any]] = []
    catalog = catalog_items.select(item_col).with_row_index("_catalog_row")
    expected_columns: list[str] | None = None
    for uid in users[uid_col].to_list():
        pairs = catalog.with_columns(pl.lit(uid).alias(uid_col)).select(
            uid_col,
            item_col,
        )
        assigned = user_sid_pipeline.update(pairs)
        columns = validate_user_sid_output(pairs, assigned)
        if expected_columns is None:
            _validate_codebook_sizes(columns, sizes)
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError("user SID columns must be identical for every user")
        aligned = (
            catalog.join(
                assigned.select(item_col, *columns),
                on=item_col,
                how="left",
                validate="1:1",
            )
            .sort("_catalog_row")
            .select(*columns)
        )
        per_user.append(
            _calculate_sid_quality_numpy(
                aligned.to_numpy(),
                sizes,
                normalized_embeddings,
            )
        )
    return _macro_average_user_quality(
        per_user,
        catalog_size=catalog_items.height,
    )
