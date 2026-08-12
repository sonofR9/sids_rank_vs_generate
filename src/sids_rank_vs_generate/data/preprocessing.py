from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl

from .protocols import (
    POSTER_FULL_33148_PROTOCOL_ID,
    THREE_WAY_PROTOCOL_ID,
    uses_legacy_heldout,
    validate_protocol_id,
)


REQUIRED_COLUMNS = ("uid", "item_id", "timestamp")
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class PreprocessingConfig:
    """Общие настройки предобработки"""

    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    timestamp_unit_seconds: float = 1.0
    history_days: float | None = None
    validation_fraction: float = 0.1
    catalog_from_fit: bool = False
    protocol_id: str = THREE_WAY_PROTOCOL_ID

    def __post_init__(self) -> None:
        validate_protocol_id(self.protocol_id)
        if self.core_min_interactions_per_item <= 0:
            raise ValueError("core_min_interactions_per_item must be positive")
        if self.history_days is not None and self.history_days <= 0:
            raise ValueError("history_days must be positive when provided")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.history_days is not None and self.history_days <= self.test_days:
            raise ValueError("history_days must be larger than test_days")
        if self.timestamp_unit_seconds <= 0:
            raise ValueError("timestamp_unit_seconds must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.protocol_id == POSTER_FULL_33148_PROTOCOL_ID:
            if self.core_min_interactions_per_item != 5:
                raise ValueError("poster protocol requires item core 5")
            if self.history_days is not None:
                raise ValueError("poster protocol requires the full history")
            if self.test_days != 7.0:
                raise ValueError("poster protocol requires a 7-day heldout")
            if self.timestamp_unit_seconds != 5.0:
                raise ValueError("poster protocol requires 5-second timestamps")
            if self.catalog_from_fit:
                raise ValueError("poster protocol requires a full-data catalog")


@dataclass(frozen=True)
class PreparedInteractions:
    """Подготовленные данные и временные границы выборок"""

    data: pl.DataFrame
    train: pl.DataFrame
    test: pl.DataFrame
    item_mapping: pl.DataFrame
    catalog_size: int
    test_start_timestamp: Any
    timestamp_unit_seconds: float
    fit: pl.DataFrame
    validation: pl.DataFrame
    validation_start_timestamp: Any
    protocol_id: str = THREE_WAY_PROTOCOL_ID

    def summary(self) -> dict[str, int | float | str | bool]:
        return {
            "interactions": self.data.height,
            "train_interactions": self.train.height,
            "fit_interactions": self.fit.height,
            "validation_interactions": self.validation.height,
            "test_interactions": self.test.height,
            "users": self.data["uid"].n_unique(),
            "catalog_size": self.catalog_size,
            "test_start_timestamp": str(self.test_start_timestamp),
            "validation_start_timestamp": str(self.validation_start_timestamp),
            "fit_fingerprint": interaction_fingerprint(self.fit),
            "protocol_id": self.protocol_id,
            "validation_reuses_test": uses_legacy_heldout(self.protocol_id),
        }


def interaction_fingerprint(frame: pl.DataFrame) -> str:
    """Стабильный хеш строк обучающей выборки"""

    canonical = frame.select(REQUIRED_COLUMNS).sort(["timestamp", "uid", "item_id"])
    digest = sha256()
    digest.update(str(canonical.schema).encode("utf-8"))
    for batch in canonical.iter_slices(n_rows=65_536):
        digest.update(batch.hash_rows(seed=42).to_numpy().tobytes())
    return digest.hexdigest()


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
    """Загрузить основные поля взаимодействий"""

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


def timestamp_days_before(
    timestamp: Any,
    timestamp_dtype: pl.DataType,
    days: float,
    timestamp_unit_seconds: float,
) -> Any:
    """Move a temporal or binned numeric timestamp back by ``days``."""

    if timestamp_dtype.is_temporal():
        return timestamp - timedelta(days=days)
    return timestamp - days * SECONDS_PER_DAY / timestamp_unit_seconds


def _validation_start_timestamp(
    train_pool: pl.DataFrame,
    validation_fraction: float,
    test_start_timestamp: Any,
) -> Any:
    if train_pool.height < 2 or train_pool["timestamp"].n_unique() < 2:
        return test_start_timestamp
    return train_pool["timestamp"].quantile(
        1.0 - validation_fraction,
        interpolation="nearest",
    )


def prepare_interactions(
    interactions: pl.DataFrame,
    config: PreprocessingConfig | None = None,
) -> PreparedInteractions:
    """Отфильтровать и разделить данные по времени"""

    config = config or PreprocessingConfig()
    _validate_interactions(interactions)
    source = interactions.select(REQUIRED_COLUMNS)
    if config.history_days is not None:
        history_start = timestamp_days_before(
            source["timestamp"].max(),
            source.schema["timestamp"],
            config.history_days,
            config.timestamp_unit_seconds,
        )
        source = source.filter(pl.col("timestamp") >= history_start)
    source = source.sort(["timestamp", "uid"])

    boundary_source = source
    if uses_legacy_heldout(config.protocol_id):
        # Старый poster считал границу после item-core
        boundary_items = (
            source.group_by("item_id")
            .len(name="interaction_count")
            .filter(
                pl.col("interaction_count")
                >= config.core_min_interactions_per_item
            )
            .select("item_id")
        )
        boundary_source = source.join(
            boundary_items,
            on="item_id",
            how="inner",
        )
        if boundary_source.is_empty():
            raise ValueError("no interactions remain after item-core filtering")

    test_start = _test_start_timestamp(
        boundary_source,
        config.test_days,
        config.timestamp_unit_seconds,
    )
    source_train = source.filter(pl.col("timestamp") < test_start)
    if uses_legacy_heldout(config.protocol_id):
        validation_start = test_start
        source_fit = source_train
    else:
        validation_start = _validation_start_timestamp(
            source_train,
            config.validation_fraction,
            test_start,
        )
        source_fit = source_train.filter(pl.col("timestamp") < validation_start)
        if source_fit.is_empty():
            # В маленьких тестах validation может отсутствовать
            validation_start = test_start
            source_fit = source_train

    catalog_source = source_fit if config.catalog_from_fit else source
    eligible_items = (
        catalog_source.group_by("item_id")
        .len(name="interaction_count")
        .filter(pl.col("interaction_count") >= config.core_min_interactions_per_item)
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

    train = data.filter(pl.col("timestamp") < test_start)
    fit = data.filter(pl.col("timestamp") < validation_start)
    test = data.filter(pl.col("timestamp") >= test_start)
    validation = (
        test
        if uses_legacy_heldout(config.protocol_id)
        else data.filter(
            (pl.col("timestamp") >= validation_start)
            & (pl.col("timestamp") < test_start)
        )
    )

    return PreparedInteractions(
        data=data,
        train=train,
        test=test,
        item_mapping=item_mapping,
        catalog_size=item_mapping.height,
        test_start_timestamp=test_start,
        timestamp_unit_seconds=config.timestamp_unit_seconds,
        fit=fit,
        validation=validation,
        validation_start_timestamp=validation_start,
        protocol_id=config.protocol_id,
    )


def load_and_prepare_interactions(
    path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreparedInteractions:
    return prepare_interactions(load_interactions(path), config=config)


def build_histories(frame: pl.DataFrame) -> dict[Any, list[int]]:
    """Построить истории айтемов по времени"""

    _validate_interactions(frame)
    grouped = (
        frame.sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"))
    )
    return {row["uid"]: list(row["item_id"]) for row in grouped.iter_rows(named=True)}


def build_targets(frame: pl.DataFrame) -> dict[Any, list[int]]:
    return build_histories(frame)
