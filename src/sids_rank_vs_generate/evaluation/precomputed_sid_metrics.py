from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sids_rank_vs_generate.evaluation.collision_quality import (
    adjusted_semantic_variance_explained,
    codebook_quality,
    infer_codebook_sizes,
    intra_code_similarity_v2,
    prefix_coherence,
    sid_distribution_quality,
)
from sids_rank_vs_generate.evaluation.sid_statistics import calculate_sid_statistics
from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


@dataclass(frozen=True)
class PrecomputedSidMetricsConfig:
    sids_dir: Path
    embeddings_path: Path
    output_path: Path | None = None
    recursive: bool = False
    codebook_sizes: tuple[int, ...] | None = None


def _validate_sids(sids: pl.DataFrame, *, source: str) -> list[str]:
    if sids.is_empty():
        raise ValueError(f"{source} must not be empty")
    if "item_id" not in sids.columns:
        raise ValueError(f"{source} must contain item_id")
    if sids["item_id"].null_count():
        raise ValueError(f"{source} contains null item_id values")
    if sids["item_id"].n_unique() != sids.height:
        raise ValueError(f"{source} contains duplicate item_id values")

    malformed = [
        column
        for column in sids.columns
        if column.startswith("sid_") and not column[4:].isdigit()
    ]
    if malformed:
        raise ValueError(
            f"{source} contains malformed SID columns: {', '.join(malformed)}"
        )
    sid_cols = get_sid_cols(sids)
    if not sid_cols:
        raise ValueError(f"{source} has no sid_<level> columns")
    for column in sid_cols:
        if sids[column].null_count():
            raise ValueError(f"{source} contains nulls in {column}")
        if not sids.schema[column].is_integer():
            raise TypeError(f"{source} column {column} must contain integer codes")
    return sid_cols


def _load_embeddings(path: Path) -> tuple[pl.DataFrame, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"embeddings parquet not found: {resolved}")
    embeddings = pl.read_parquet(resolved)
    if embeddings.is_empty():
        raise ValueError("embeddings parquet must not be empty")
    if "item_id" not in embeddings.columns:
        raise ValueError("embeddings parquet must contain item_id")
    embedding_col = (
        "normalized_embed"
        if "normalized_embed" in embeddings.columns
        else "embed" if "embed" in embeddings.columns else None
    )
    if embedding_col is None:
        raise ValueError(
            "embeddings parquet must contain normalized_embed or embed"
        )
    if embeddings["item_id"].null_count():
        raise ValueError("embeddings parquet contains null item_id values")
    if embeddings["item_id"].n_unique() != embeddings.height:
        raise ValueError("embeddings parquet contains duplicate item_id values")
    return embeddings.select("item_id", embedding_col), embedding_col


def calculate_all_sid_metrics(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    embedding_col: str | None = None,
    codebook_sizes: Sequence[int] | None = None,
    source: str = "semantic IDs",
) -> dict[str, Any]:
    """Calculate every common SID metric for an existing assignment frame."""

    sid_cols = _validate_sids(sids, source=source)
    selected_embedding_col = embedding_col or (
        "normalized_embed"
        if "normalized_embed" in embeddings.columns
        else "embed" if "embed" in embeddings.columns else None
    )
    if selected_embedding_col is None:
        raise ValueError("embeddings must contain normalized_embed or embed")

    matched_items = sids.select("item_id").join(
        embeddings.select("item_id"),
        on="item_id",
        how="inner",
    )
    if matched_items.height != sids.height:
        raise ValueError(
            f"{source} has {sids.height - matched_items.height} items without embeddings"
        )

    sizes = (
        tuple(int(size) for size in codebook_sizes)
        if codebook_sizes is not None
        else infer_codebook_sizes(None, sids)
    )
    metrics = calculate_sid_statistics(sids.select("item_id", *sid_cols))
    coherence_metrics = prefix_coherence(
        sids,
        embeddings,
        embedding_col=selected_embedding_col,
    )
    metrics.update(coherence_metrics)
    metrics["path_average_similarity"] = coherence_metrics[
        f"prefix_coherence_at_{len(sid_cols)}"
    ]
    metrics["intra_code_similarity_v2"] = intra_code_similarity_v2(
        sids,
        embeddings,
        embedding_col=selected_embedding_col,
    )
    metrics.update(
        adjusted_semantic_variance_explained(
            sids,
            embeddings,
            embedding_col=selected_embedding_col,
        )
    )
    metrics.update(sid_distribution_quality(sids, sizes))
    quality = codebook_quality(sids, sizes)
    metrics["codebook_quality"] = quality

    path_counts = sids.group_by(sid_cols).len(name="count")["count"].to_numpy()
    probabilities = path_counts.astype(np.float64) / path_counts.sum()
    sorted_counts = np.sort(path_counts.astype(np.float64))
    positions = np.arange(1, len(sorted_counts) + 1, dtype=np.float64)
    colliding_counts = path_counts[path_counts > 1]
    metrics.update(
        {
            "p95_collision_group_size": (
                float(np.quantile(colliding_counts, 0.95, method="nearest"))
                if len(colliding_counts)
                else 0.0
            ),
            "cur_total": quality["codebook_utilization_rate_total"],
            "code_gini": float(
                2 * np.dot(positions, sorted_counts)
                / (len(sorted_counts) * sorted_counts.sum())
                - (len(sorted_counts) + 1) / len(sorted_counts)
            ),
            "code_entropy": float(-np.sum(probabilities * np.log(probabilities))),
            "intra_code_similarity": metrics["path_average_similarity"],
            "valid_sid_rate": 1.0,
        }
    )
    return metrics


def run_precomputed_sid_metrics(
    config: PrecomputedSidMetricsConfig,
) -> dict[str, Any]:
    """Measure every SID parquet in a directory without training a pipeline."""

    sids_dir = config.sids_dir.expanduser().resolve()
    if not sids_dir.is_dir():
        raise NotADirectoryError(f"SID parquet directory not found: {sids_dir}")
    embeddings_path = config.embeddings_path.expanduser().resolve()
    embeddings, embedding_col = _load_embeddings(embeddings_path)

    candidates = (
        sids_dir.rglob("*.parquet")
        if config.recursive
        else sids_dir.glob("*.parquet")
    )
    sid_paths = sorted(
        path for path in candidates if path.resolve() != embeddings_path
    )
    if not sid_paths:
        scan_kind = "recursively" if config.recursive else ""
        raise FileNotFoundError(
            f"no SID parquet files found {scan_kind} in {sids_dir}".replace("  ", " ")
        )

    files: dict[str, Any] = {}
    for sid_path in sid_paths:
        relative_path = sid_path.relative_to(sids_dir).as_posix()
        sids = pl.read_parquet(sid_path)
        sid_cols = _validate_sids(sids, source=relative_path)
        codebook_sizes = (
            config.codebook_sizes
            if config.codebook_sizes is not None
            else infer_codebook_sizes(None, sids)
        )
        files[relative_path] = {
            "sid_columns": sid_cols,
            "codebook_sizes": list(codebook_sizes),
            "metrics": calculate_all_sid_metrics(
                sids,
                embeddings,
                embedding_col=embedding_col,
                codebook_sizes=codebook_sizes,
                source=relative_path,
            ),
        }

    output_path = (
        config.output_path.expanduser().resolve()
        if config.output_path is not None
        else sids_dir / "sid_metrics.json"
    )
    result: dict[str, Any] = {
        "mode": "precomputed_sid_metrics",
        "sids_directory": str(sids_dir),
        "embeddings": str(embeddings_path),
        "embedding_column": embedding_col,
        "recursive": config.recursive,
        "codebook_sizes": (
            list(config.codebook_sizes)
            if config.codebook_sizes is not None
            else None
        ),
        "file_count": len(files),
        "files": files,
        "output": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
