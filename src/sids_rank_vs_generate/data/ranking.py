from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..datasets import DatasetFiles
from ..sids.user_aware import IUserAwareSidPipeline, user_sid_columns
from .preprocessing import PreparedInteractions


DENSE_COLUMNS = (
    "user_lag_listen_cnt",
    "user_lag_like_cnt",
    "user_lag_full_play_cnt",
    "user_lag_skip_cnt",
    "item_lag_listen_cnt",
    "item_lag_like_cnt",
    "item_lag_full_play_cnt",
    "item_lag_skip_cnt",
    "ui_lag_listen_cnt",
    "ui_lag_like_cnt",
    "ui_lag_full_play_cnt",
    "ui_lag_skip_cnt",
    "user_lag_avg_played_ratio",
    "item_lag_avg_played_ratio",
    "ui_lag_avg_played_ratio",
)
MULTIVALENT_COLUMNS = ("artist_ids", "album_ids")
HASH_CARDINALITY = 65_536
NUM_HASHES = 2


@dataclass(frozen=True)
class RankingData:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    dense_train: pl.DataFrame
    sid_columns: tuple[str, ...]
    num_users: int
    num_history_items: int


def _require_file(path: Path | None, description: str) -> Path:
    if path is None:
        raise ValueError(f"{description} is required for the ranking stage")
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _sid_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    columns = (
        column
        for column in frame.columns
        if column.startswith("sid_") and column[4:].isdigit()
    )
    return tuple(sorted(columns, key=lambda column: int(column[4:])))


def _history_snapshots(
    prepared: PreparedInteractions,
    history_item_mapping: pl.DataFrame,
    max_history_len: int,
) -> pl.DataFrame:
    item_history = pl.concat_list(
        [
            pl.col("history_item_idx").shift(offset).over("uid")
            for offset in range(max_history_len - 1, -1, -1)
        ]
    ).list.drop_nulls()
    time_history = pl.concat_list(
        [
            pl.col("timestamp").shift(offset).over("uid")
            for offset in range(max_history_len - 1, -1, -1)
        ]
    ).list.drop_nulls()
    return (
        prepared.data.join(history_item_mapping, on="item_id", how="inner")
        .sort(["uid", "timestamp"])
        .with_columns(
            item_history.alias("history_item_idxs"),
            time_history.alias("history_timestamps"),
        )
        .select(
            "uid",
            pl.col("timestamp").alias("history_timestamp"),
            "history_item_idxs",
            "history_timestamps",
        )
    )


def _hash_expression(expression: pl.Expr, salt: int) -> pl.Expr:
    return (
        (expression.cast(pl.Int64) * 2_654_435_761 + salt) % (HASH_CARDINALITY - 1)
    ) + 1


def _add_hashed_features(
    frame: pl.DataFrame,
    sid_columns: tuple[str, ...],
) -> pl.DataFrame:
    expressions = [_hash_expression(pl.col("uid"), 17).alias("uid_hash")]
    expressions.extend(
        _hash_expression(pl.col(column), 1_009 + level * 9_973).alias(f"{column}_hash")
        for level, column in enumerate(sid_columns)
    )
    for column, salt in (("artist_ids", 20_011), ("album_ids", 40_009)):
        expressions.extend(
            pl.col(column)
            .list.eval(_hash_expression(pl.element(), salt + hash_index * 104_729))
            .alias(f"{column}_hash_{hash_index}")
            for hash_index in range(NUM_HASHES)
        )
    return frame.with_columns(expressions)


def _load_multivalent_mapping(
    path: Path,
    id_column: str,
    eligible_items: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .join(eligible_items, on="item_id", how="inner")
        .group_by("item_id")
        .agg(pl.col(id_column).alias(f"{id_column}s"))
    )


def _load_dcn_candidates(
    dataset: DatasetFiles,
    prepared: PreparedInteractions,
    items_with_sids: pl.DataFrame,
    max_rows: int | None,
) -> pl.DataFrame:
    ranking_path = _require_file(
        dataset.ranking_interactions_path,
        "DCNv2 lag-feature interactions parquet",
    )
    artist_path = _require_file(dataset.artist_mapping_path, "artist mapping parquet")
    album_path = _require_file(dataset.album_mapping_path, "album mapping parquet")

    item_mapping = prepared.item_mapping.lazy()
    eligible_items = item_mapping.select(
        pl.col("original_item_id").cast(pl.UInt32).alias("item_id")
    )
    candidates = pl.scan_parquet(ranking_path)
    required_columns = set(DENSE_COLUMNS) | {
        "uid",
        "timestamp",
        "item_id",
        "track_length_seconds",
        "played_ratio_pct",
        "is_like",
    }
    missing = required_columns - set(candidates.collect_schema().names())
    if missing:
        raise ValueError(f"ranking interactions are missing columns: {sorted(missing)}")

    candidates = candidates.join(eligible_items, on="item_id", how="inner")
    if max_rows is not None:
        candidates = candidates.top_k(max_rows, by="timestamp")

    artist_mapping = _load_multivalent_mapping(artist_path, "artist_id", eligible_items)
    album_mapping = _load_multivalent_mapping(album_path, "album_id", eligible_items)
    sids = (
        items_with_sids.select("item_id", *_sid_columns(items_with_sids))
        .rename({"item_id": "mapped_item_id"})
        .lazy()
    )

    return (
        candidates.join(artist_mapping, on="item_id", how="left")
        .join(album_mapping, on="item_id", how="left")
        .rename({"item_id": "original_item_id"})
        .join(item_mapping, on="original_item_id", how="inner")
        .join(sids, left_on="item_id", right_on="mapped_item_id", how="inner")
        .with_columns(
            pl.col("artist_ids").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
            pl.col("album_ids").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
            pl.col("is_like").cast(pl.Float32).alias("target"),
            pl.col(DENSE_COLUMNS).cast(pl.Float32).fill_nan(0.0).fill_null(0.0),
        )
        .collect()
    )


def prepare_ranking_data(
    prepared: PreparedInteractions,
    items_with_sids: pl.DataFrame,
    dataset: DatasetFiles,
    *,
    max_history_len: int,
    min_history_len: int,
    validation_fraction: float,
    max_rows: int | None,
    user_sid_pipeline: IUserAwareSidPipeline | None = None,
) -> RankingData:
    """Добавить признаки и историю лайков к кандидатам"""

    del validation_fraction  # Граница задается общей предобработкой

    sid_columns = _sid_columns(items_with_sids)
    if not sid_columns:
        raise ValueError("ranking requires at least one semantic-ID column")
    sid_frame = items_with_sids.select("item_id", *sid_columns).sort("item_id")
    if sid_frame.height != prepared.catalog_size:
        raise ValueError("ranking requires SIDs for the full prepared catalog")

    sid_paths = (
        sid_frame.select(sid_columns)
        .unique()
        .sort(sid_columns)
        .with_row_index("history_item_idx", offset=1)
    )
    history_item_mapping = sid_frame.join(
        sid_paths,
        on=sid_columns,
        how="inner",
    ).select("item_id", "history_item_idx")
    candidates = (
        _load_dcn_candidates(dataset, prepared, items_with_sids, max_rows)
        .join(sid_paths, on=sid_columns, how="inner")
        .sort(["uid", "timestamp"])
    )
    candidate_sid_columns = sid_columns
    if user_sid_pipeline is not None:
        candidate_pairs = (
            candidates.select("uid", "item_id").unique().sort(["uid", "item_id"])
        )
        candidate_user_sids = user_sid_pipeline.update(candidate_pairs)
        candidate_sid_columns = tuple(user_sid_columns(candidate_user_sids))
        candidates = candidates.join(
            candidate_user_sids,
            on=["uid", "item_id"],
            how="inner",
        )
    histories = _history_snapshots(
        prepared,
        history_item_mapping,
        max_history_len,
    )
    merged = candidates.join_asof(
        histories,
        left_on="timestamp",
        right_on="history_timestamp",
        by="uid",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    ).filter(pl.col("history_item_idxs").list.len() >= min_history_len)
    if merged.is_empty():
        raise ValueError("ranking data has no candidates with a preceding like history")

    user_mapping = merged.select("uid").unique().sort("uid").with_row_index("user_idx")
    merged = _add_hashed_features(
        merged.join(user_mapping, on="uid", how="inner"),
        candidate_sid_columns,
    )

    train = merged.filter(pl.col("timestamp") < prepared.validation_start_timestamp)
    validation = merged.filter(
        (pl.col("timestamp") >= prepared.validation_start_timestamp)
        & (pl.col("timestamp") < prepared.test_start_timestamp)
    )
    test = merged.filter(pl.col("timestamp") >= prepared.test_start_timestamp)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.is_empty():
            raise ValueError(f"ranking {name} split is empty")
        if frame["target"].n_unique() < 2:
            raise ValueError(f"ranking {name} split must contain both labels")

    return RankingData(
        train=train,
        validation=validation,
        test=test,
        dense_train=train.select(DENSE_COLUMNS),
        sid_columns=candidate_sid_columns,
        num_users=user_mapping.height,
        num_history_items=sid_paths.height + 1,
    )


class PolarsRankingDataset(Dataset):
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.frame.row(index, named=True)


def _collate_multivalent(
    batch: list[dict[str, Any]],
    column: str,
) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(row[column]) for row in batch], dtype=torch.long)
    values = torch.tensor(
        [
            [row[f"{column}_hash_{index}"][position] for index in range(NUM_HASHES)]
            for row in batch
            for position in range(len(row[column]))
        ],
        dtype=torch.long,
    ).reshape(-1, NUM_HASHES)
    return {"values": values, "lengths": lengths}


def collate_ranking(batch: list[dict[str, Any]]) -> dict[str, Any]:
    history_lengths = torch.tensor(
        [len(row["history_item_idxs"]) for row in batch], dtype=torch.long
    )
    history_items = torch.tensor(
        [item for row in batch for item in row["history_item_idxs"]],
        dtype=torch.long,
    )
    history_deltas = (
        torch.tensor(
            [
                float(row["timestamp"] - timestamp)
                for row in batch
                for timestamp in row["history_timestamps"]
            ],
            dtype=torch.float32,
        )
        .clamp_min_(0.0)
        .log1p_()
    )
    sid_columns = tuple(
        sorted(
            (
                column
                for column in batch[0]
                if (
                    (column.startswith("sid_") and column[4:].isdigit())
                    or (column.startswith("usid_") and column[5:].isdigit())
                )
                and f"{column}_hash" in batch[0]
            ),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
    )
    sparse = {"uid": torch.tensor([row["uid_hash"] for row in batch], dtype=torch.long)}
    for column in sid_columns:
        sparse[column] = torch.tensor(
            [row[f"{column}_hash"] for row in batch],
            dtype=torch.long,
        )

    zeros = torch.zeros(history_items.numel(), dtype=torch.float32)
    candidate_track_length = torch.tensor(
        [row["track_length_seconds"] for row in batch], dtype=torch.float32
    )
    return {
        "dense_features": torch.tensor(
            [[row[column] for column in DENSE_COLUMNS] for row in batch],
            dtype=torch.float32,
        ),
        "sparse_features": sparse,
        "multivalent_features": {
            "artist_ids": _collate_multivalent(batch, "artist_ids"),
            "album_ids": _collate_multivalent(batch, "album_ids"),
        },
        "user_idx": torch.tensor([row["user_idx"] for row in batch], dtype=torch.long),
        "candidate_item_idx": torch.tensor(
            [row["history_item_idx"] for row in batch], dtype=torch.long
        ),
        "candidate_timestamp": torch.tensor(
            [row["timestamp"] for row in batch], dtype=torch.float64
        ),
        "candidate_is_organic": torch.zeros(len(batch), dtype=torch.float32),
        "candidate_track_length_z": (
            candidate_track_length - candidate_track_length.mean()
        )
        / candidate_track_length.std().clamp_min(1.0),
        "candidate_has_track_length": (candidate_track_length > 0).float(),
        "history_item_idxs_flat": history_items,
        "history_event_type_idxs_flat": torch.full_like(history_items, 2),
        "history_is_organic_flat": zeros,
        "history_played_ratio_flat": zeros,
        "history_has_played_ratio_flat": zeros,
        "history_track_length_z_flat": zeros,
        "history_has_track_length_flat": zeros,
        "history_time_delta_z_flat": history_deltas,
        "history_lengths": history_lengths,
        "target": torch.tensor([row["target"] for row in batch], dtype=torch.float32),
    }


def build_ranking_loaders(
    data: RankingData,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_ranking,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(PolarsRankingDataset(data.train), shuffle=True, **common),
        DataLoader(PolarsRankingDataset(data.validation), shuffle=False, **common),
        DataLoader(PolarsRankingDataset(data.test), shuffle=False, **common),
    )
