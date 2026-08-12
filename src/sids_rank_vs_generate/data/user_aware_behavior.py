from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl

from .preprocessing import PreparedInteractions


SOURCE_COLUMNS = ("uid", "item_id", "timestamp", "is_like")
OUTPUT_COLUMNS = ("uid", "item_id", "timestamp", "target")


@dataclass(frozen=True)
class UserAwareBehaviorSummary:
    """Сводка fit-only поведенческих меток"""

    rows: int
    users: int
    pairs: int
    positive_rate: float
    min_timestamp: Any
    max_timestamp: Any
    fit_start_timestamp: Any
    fit_end_timestamp_exclusive: Any
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserAwareBehaviorData:
    """Поведенческие метки и их воспроизводимая сводка"""

    frame: pl.DataFrame
    summary: UserAwareBehaviorSummary


def user_aware_behavior_fingerprint(frame: pl.DataFrame) -> str:
    """Посчитать стабильный хеш строк вместе с target"""

    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"behavior frame is missing columns: {missing}")
    canonical = frame.select(OUTPUT_COLUMNS).sort(
        ["timestamp", "uid", "item_id", "target"]
    )
    digest = sha256(b"user-aware-fit-behavior-v1\0")
    digest.update(str(canonical.schema).encode("utf-8"))
    for batch in canonical.iter_slices(n_rows=65_536):
        digest.update(batch.hash_rows(seed=42).to_numpy().tobytes())
    return digest.hexdigest()


def _require_source(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"ranking interactions parquet not found: {resolved}")
    return resolved


def _validate_mapping(prepared: PreparedInteractions) -> pl.DataFrame:
    required = {"original_item_id", "item_id"}
    missing = required - set(prepared.item_mapping.columns)
    if missing:
        raise ValueError(f"prepared item mapping is missing columns: {sorted(missing)}")
    mapping = prepared.item_mapping.select("original_item_id", "item_id")
    if mapping.is_empty():
        raise ValueError("prepared item mapping must not be empty")
    if mapping.null_count().row(0) != (0, 0):
        raise ValueError("prepared item mapping contains nulls")
    if (
        mapping["original_item_id"].n_unique() != mapping.height
        or mapping["item_id"].n_unique() != mapping.height
    ):
        raise ValueError("prepared item mapping must be one-to-one")
    return mapping


def _validate_fit_source_nulls(fit_source: pl.LazyFrame) -> None:
    null_counts = fit_source.select(
        pl.col(column).null_count().alias(column) for column in SOURCE_COLUMNS
    ).collect().row(0, named=True)
    columns_with_nulls = [column for column, count in null_counts.items() if count]
    if columns_with_nulls:
        raise ValueError(f"fit behavior columns contain nulls: {columns_with_nulls}")


def _validate_targets(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise ValueError("no fit behavior rows remain after catalog filtering")
    if frame["target"].null_count():
        raise ValueError("fit behavior target contains nulls")
    labels = set(frame["target"].unique().to_list())
    if not labels.issubset({0.0, 1.0}):
        raise ValueError("fit behavior target must be binary")
    if labels != {0.0, 1.0}:
        raise ValueError("fit behavior must contain both labels 0 and 1")


def load_fit_user_aware_behavior(
    ranking_interactions_path: str | Path,
    prepared: PreparedInteractions,
    *,
    max_rows: int | None = None,
) -> UserAwareBehaviorData:
    """Загрузить только fit-метки для user-aware 2.0"""

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if prepared.fit.is_empty() or "timestamp" not in prepared.fit.columns:
        raise ValueError("prepared fit interactions must contain timestamps")

    source_path = _require_source(ranking_interactions_path)
    mapping = _validate_mapping(prepared)
    source = pl.scan_parquet(source_path)
    missing = set(SOURCE_COLUMNS) - set(source.collect_schema().names())
    if missing:
        raise ValueError(f"ranking interactions are missing columns: {sorted(missing)}")

    selected = source.select(SOURCE_COLUMNS)
    timestamp_nulls = selected.select(pl.col("timestamp").null_count()).collect().item()
    if timestamp_nulls:
        raise ValueError("ranking interaction timestamp contains nulls")

    fit_start_timestamp = prepared.fit["timestamp"].min()
    fit_source = selected.filter(
        (pl.col("timestamp") >= pl.lit(fit_start_timestamp))
        & (pl.col("timestamp") < pl.lit(prepared.validation_start_timestamp))
    )
    _validate_fit_source_nulls(fit_source)

    original_item_dtype = mapping.schema["original_item_id"]
    mapped = (
        fit_source.with_columns(
            pl.col("item_id").cast(original_item_dtype, strict=True)
        )
        .rename({"item_id": "original_item_id"})
        .join(
            mapping.rename({"item_id": "mapped_item_id"}).lazy(),
            on="original_item_id",
            how="inner",
        )
        .select(
            "uid",
            pl.col("mapped_item_id").cast(pl.Int64).alias("item_id"),
            "timestamp",
            pl.col("is_like").cast(pl.Float32, strict=True).alias("target"),
        )
        .sort(["timestamp", "uid", "item_id", "target"])
    )
    if max_rows is not None:
        mapped = mapped.tail(max_rows)
    frame = mapped.collect().sort(["timestamp", "uid", "item_id", "target"])
    _validate_targets(frame)

    summary = UserAwareBehaviorSummary(
        rows=frame.height,
        users=frame["uid"].n_unique(),
        pairs=frame.select(pl.struct("uid", "item_id").n_unique()).item(),
        positive_rate=float(frame["target"].mean()),
        min_timestamp=frame["timestamp"].min(),
        max_timestamp=frame["timestamp"].max(),
        fit_start_timestamp=fit_start_timestamp,
        fit_end_timestamp_exclusive=prepared.validation_start_timestamp,
        fingerprint=user_aware_behavior_fingerprint(frame),
    )
    return UserAwareBehaviorData(frame=frame, summary=summary)
