from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


DATASET_SIZES = ("50m",)
DEFAULT_DATASET_SIZE = "50m"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "sids_rank_vs_generate" / "datasets"
HF_DATASET_REPO = "yandex/yambda"
HF_LIKES_FILE = "flat/50m/likes.parquet"
HF_EMBEDDINGS_FILE = "embeddings.parquet"
PREPARED_EMBEDDINGS_FILE = "embeddings.parquet"


@dataclass(frozen=True)
class DatasetFiles:
    """Resolved Yambda-50M likes and its prepared content vectors."""

    size: str
    interactions_path: Path
    embeddings_path: Path


def _hf_download(repo_id: str, *, filename: str, repo_type: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Yambda-50M download requires huggingface_hub; install the project "
            "dependencies with `pip install -e .`"
        ) from error
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )


def prepare_embedding_subset(
    interactions_path: str | Path,
    source_embeddings_path: str | Path,
    destination_path: str | Path,
) -> Path:
    """Filter global audio vectors to Yambda-50M likes and L2-normalize them."""

    interactions_path = Path(interactions_path).expanduser()
    source_embeddings_path = Path(source_embeddings_path).expanduser()
    destination_path = Path(destination_path).expanduser()
    if not interactions_path.is_file():
        raise FileNotFoundError(f"likes parquet not found: {interactions_path}")
    if not source_embeddings_path.is_file():
        raise FileNotFoundError(
            f"source embeddings parquet not found: {source_embeddings_path}"
        )

    source_query = pl.scan_parquet(source_embeddings_path)
    schema = source_query.collect_schema()
    if "item_id" not in schema:
        raise ValueError("source embeddings must contain item_id")
    vector_column = (
        "normalized_embed"
        if "normalized_embed" in schema
        else "embed" if "embed" in schema else None
    )
    if vector_column is None:
        raise ValueError(
            "source embeddings must contain embed or normalized_embed"
        )

    liked_items = (
        pl.scan_parquet(interactions_path)
        .select("item_id")
        .unique()
    )
    subset = (
        source_query.select("item_id", vector_column)
        .join(liked_items, on="item_id", how="inner")
        .sort("item_id")
        .collect()
    )
    if subset.is_empty():
        raise ValueError("no Yambda-50M liked items have content embeddings")
    first_vector = subset[vector_column][0]
    if first_vector is None or len(first_vector) == 0:
        raise ValueError("content embedding vectors must not be empty")
    width = len(first_vector)
    vector_dtype = subset.schema[vector_column]
    if isinstance(vector_dtype, pl.Array):
        matrix = subset[vector_column].to_numpy()
    else:
        matrix = subset[vector_column].list.to_array(width).to_numpy()
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("content embeddings contain zero or non-finite vectors")
    normalized = matrix / norms

    prepared = pl.DataFrame(
        {
            "item_id": subset["item_id"],
            "normalized_embed": pl.Series(
                "normalized_embed",
                normalized,
                dtype=pl.Array(pl.Float32, width),
            ),
        }
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_parquet(destination_path)
    return destination_path


def resolve_dataset(
    size: str = DEFAULT_DATASET_SIZE,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    source_embeddings_path: Path | None = None,
    downloader: Callable[..., str] | None = None,
) -> DatasetFiles:
    """Resolve the official Yambda-50M likes and a compact embedding subset."""

    if size != DEFAULT_DATASET_SIZE:
        raise ValueError("only Yambda-50M is supported")
    download = downloader or _hf_download
    interactions_path = Path(
        download(HF_DATASET_REPO, filename=HF_LIKES_FILE, repo_type="dataset")
    ).expanduser()

    dataset_dir = cache_root.expanduser() / "yambda-50m"
    embeddings_path = dataset_dir / PREPARED_EMBEDDINGS_FILE
    if not embeddings_path.is_file():
        if source_embeddings_path is not None:
            source = Path(source_embeddings_path).expanduser()
        elif Path("embeddings.parquet").is_file():
            source = Path("embeddings.parquet")
        else:
            source = Path(
                download(
                    HF_DATASET_REPO,
                    filename=HF_EMBEDDINGS_FILE,
                    repo_type="dataset",
                )
            ).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                "Yambda embeddings are missing: "
                f"{source}. Pass source_embeddings_path explicitly."
            )
        prepare_embedding_subset(
            interactions_path,
            source,
            embeddings_path,
        )

    return DatasetFiles(
        size=DEFAULT_DATASET_SIZE,
        interactions_path=interactions_path,
        embeddings_path=embeddings_path,
    )
