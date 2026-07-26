from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl


REQUIRED_COLUMNS = ("uid", "item_id", "timestamp")
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters shared by training and evaluation experiments."""

    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    timestamp_unit_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.core_min_interactions_per_item <= 0:
            raise ValueError("core_min_interactions_per_item must be positive")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.timestamp_unit_seconds <= 0:
            raise ValueError("timestamp_unit_seconds must be positive")


@dataclass(frozen=True)
class PreparedInteractions:
    """Dense item mapping and one reusable chronological train/test split."""

    data: pl.DataFrame
    train: pl.DataFrame
    test: pl.DataFrame
    item_mapping: pl.DataFrame
    catalog_size: int
    test_start_timestamp: Any

    def summary(self) -> dict[str, int | float | str]:
        return {
            "interactions": self.data.height,
            "train_interactions": self.train.height,
            "test_interactions": self.test.height,
            "users": self.data["uid"].n_unique(),
            "catalog_size": self.catalog_size,
            "test_start_timestamp": str(self.test_start_timestamp),
        }


def _validate_interactions(data: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"missing required interaction columns: {missing}")
    if data.is_empty():
        raise ValueError("interactions must not be empty")

    null_counts = data.select(
        pl.col(column).null_count().alias(column) for column in REQUIRED_COLUMNS
    ).row(0, named=True)
    columns_with_nulls = [column for column, count in null_counts.items() if count]
    if columns_with_nulls:
        raise ValueError(f"interaction columns contain nulls: {columns_with_nulls}")


def load_interactions(
    path: str | Path,
    *,
    latest_interactions: int | None = None,
) -> pl.DataFrame:
    """Load only the canonical interaction fields from parquet."""

    interactions_path = Path(path).expanduser()
    if not interactions_path.is_file():
        raise FileNotFoundError(f"interactions parquet not found: {interactions_path}")
    if latest_interactions is not None and latest_interactions <= 0:
        raise ValueError("latest_interactions must be positive")

    query = pl.scan_parquet(interactions_path).select(REQUIRED_COLUMNS)
    if latest_interactions is not None:
        query = query.top_k(latest_interactions, by="timestamp")
    data = query.collect()
    _validate_interactions(data)
    return data


def _test_start_timestamp(
    data: pl.DataFrame,
    test_days: float,
    timestamp_unit_seconds: float,
) -> Any:
    max_timestamp = data["timestamp"].max()
    timestamp_dtype = data.schema["timestamp"]
    if timestamp_dtype.is_temporal():
        return max_timestamp - timedelta(days=test_days)
    return max_timestamp - test_days * SECONDS_PER_DAY / timestamp_unit_seconds


def prepare_interactions(
    interactions: pl.DataFrame,
    config: PreprocessingConfig | None = None,
) -> PreparedInteractions:
    """Core-filter, map item ids once, and split the data chronologically."""

    config = config or PreprocessingConfig()
    _validate_interactions(interactions)
    source = interactions.select(REQUIRED_COLUMNS)

    eligible_items = (
        source.group_by("item_id")
        .len(name="interaction_count")
        .filter(
            pl.col("interaction_count")
            >= config.core_min_interactions_per_item
        )
        .select("item_id")
    )
    filtered = source.join(eligible_items, on="item_id", how="inner")
    if filtered.is_empty():
        raise ValueError("no interactions remain after item-core filtering")

    item_mapping = (
        filtered.select(pl.col("item_id").alias("original_item_id"))
        .unique()
        .sort("original_item_id")
        .with_row_index("item_id")
        .with_columns(pl.col("item_id").cast(pl.Int64))
        .select("original_item_id", "item_id")
    )
    data = (
        filtered.rename({"item_id": "original_item_id"})
        .join(item_mapping, on="original_item_id", how="inner")
        .select(REQUIRED_COLUMNS)
        .sort(["timestamp", "uid"])
    )

    test_start = _test_start_timestamp(
        data,
        config.test_days,
        config.timestamp_unit_seconds,
    )
    train = data.filter(pl.col("timestamp") < test_start)
    test = data.filter(pl.col("timestamp") >= test_start)

    return PreparedInteractions(
        data=data,
        train=train,
        test=test,
        item_mapping=item_mapping,
        catalog_size=item_mapping.height,
        test_start_timestamp=test_start,
    )


def load_and_prepare_interactions(
    path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreparedInteractions:
    return prepare_interactions(load_interactions(path), config=config)


def build_histories(frame: pl.DataFrame) -> dict[Any, list[int]]:
    """Build chronological item histories without repeating split logic."""

    _validate_interactions(frame)
    grouped = (
        frame.sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"))
    )
    return {
        row["uid"]: list(row["item_id"])
        for row in grouped.iter_rows(named=True)
    }


def build_targets(frame: pl.DataFrame) -> dict[Any, list[int]]:
    return build_histories(frame)
