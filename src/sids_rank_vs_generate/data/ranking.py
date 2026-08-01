from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import PreparedInteractions


@dataclass(frozen=True)
class RankingExample:
    sequence_index: int
    position: int
    user_index: int
    candidate_item: int
    timestamp: Any
    target: float


@dataclass(frozen=True)
class RankingData:
    sequences: list[tuple[list[int], list[Any]]]
    train_examples: list[RankingExample]
    validation_examples: list[RankingExample]
    test_examples: list[RankingExample]
    sid_codes: torch.Tensor
    sid_cardinalities: tuple[int, ...]
    num_users: int


class SemanticRankingDataset(Dataset):
    def __init__(
        self,
        sequences: list[tuple[list[int], list[Any]]],
        examples: list[RankingExample],
        sid_codes: torch.Tensor,
        max_history_len: int,
    ) -> None:
        self.sequences = sequences
        self.examples = examples
        self.sid_codes = sid_codes
        self.max_history_len = max_history_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        items, timestamps = self.sequences[example.sequence_index]
        start = max(0, example.position - self.max_history_len)
        history_items = items[start : example.position]
        history_timestamps = timestamps[start : example.position]
        time_deltas = [
            math.log1p(max(float(example.timestamp - timestamp), 0.0))
            for timestamp in history_timestamps
        ]
        candidate_index = example.candidate_item + 1
        return {
            "user_idx": example.user_index,
            "candidate_item_idx": candidate_index,
            "candidate_sids": self.sid_codes[example.candidate_item],
            "candidate_timestamp": example.timestamp,
            "candidate_is_organic": 0.0,
            "candidate_track_length_z": 0.0,
            "candidate_has_track_length": 0.0,
            "history_item_idxs": [item + 1 for item in history_items],
            "history_event_type_idxs": [2] * len(history_items),
            "history_is_organic": [0.0] * len(history_items),
            "history_played_ratio": [0.0] * len(history_items),
            "history_has_played_ratio": [0.0] * len(history_items),
            "history_track_length_z": [0.0] * len(history_items),
            "history_has_track_length": [0.0] * len(history_items),
            "history_time_delta_z": time_deltas,
            "target": example.target,
        }


def _flatten(batch: list[dict[str, Any]], field: str) -> list[Any]:
    return [value for example in batch for value in example[field]]


def collate_semantic_ranking(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor(
        [len(example["history_item_idxs"]) for example in batch],
        dtype=torch.long,
    )
    result = {
        "user_idx": torch.tensor([x["user_idx"] for x in batch], dtype=torch.long),
        "candidate_item_idx": torch.tensor(
            [x["candidate_item_idx"] for x in batch], dtype=torch.long
        ),
        "candidate_sids": torch.stack([x["candidate_sids"] for x in batch]),
        "candidate_timestamp": torch.tensor(
            [x["candidate_timestamp"] for x in batch], dtype=torch.float64
        ),
        "candidate_is_organic": torch.tensor(
            [x["candidate_is_organic"] for x in batch], dtype=torch.float32
        ),
        "candidate_track_length_z": torch.tensor(
            [x["candidate_track_length_z"] for x in batch], dtype=torch.float32
        ),
        "candidate_has_track_length": torch.tensor(
            [x["candidate_has_track_length"] for x in batch], dtype=torch.float32
        ),
        "history_lengths": lengths,
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.float32),
    }
    flat_fields = {
        "history_item_idxs": torch.long,
        "history_event_type_idxs": torch.long,
        "history_is_organic": torch.float32,
        "history_played_ratio": torch.float32,
        "history_has_played_ratio": torch.float32,
        "history_track_length_z": torch.float32,
        "history_has_track_length": torch.float32,
        "history_time_delta_z": torch.float32,
    }
    for field, dtype in flat_fields.items():
        result[f"{field}_flat"] = torch.tensor(_flatten(batch, field), dtype=dtype)
    return result


def _negative_item(
    rng: random.Random,
    catalog_size: int,
    positive: int,
    history: list[int],
) -> int:
    excluded = set(history)
    excluded.add(positive)
    if len(excluded) >= catalog_size:
        return (positive + 1) % catalog_size
    while True:
        candidate = rng.randrange(catalog_size)
        if candidate not in excluded:
            return candidate


def prepare_ranking_data(
    prepared: PreparedInteractions,
    items_with_sids: pl.DataFrame,
    *,
    min_history_len: int,
    negatives_per_positive: int,
    validation_fraction: float,
    max_train_examples: int | None,
    max_eval_examples: int | None,
    seed: int,
) -> RankingData:
    sid_columns = sorted(
        (column for column in items_with_sids.columns if column.startswith("sid_")),
        key=lambda column: int(column.split("_")[1]),
    )
    sid_frame = items_with_sids.select("item_id", *sid_columns).sort("item_id")
    if sid_frame.height != prepared.catalog_size:
        raise ValueError("ranking requires SIDs for the full prepared catalog")
    sid_codes = torch.tensor(
        sid_frame.select(sid_columns).to_numpy(),
        dtype=torch.long,
    )
    sid_cardinalities = tuple(
        int(sid_frame[column].max()) + 1 for column in sid_columns
    )

    grouped = (
        prepared.data.sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"), pl.col("timestamp"))
    )
    sequences: list[tuple[list[int], list[Any]]] = []
    train_examples: list[RankingExample] = []
    validation_examples: list[RankingExample] = []
    test_examples: list[RankingExample] = []
    rng = random.Random(seed)
    validation_start = prepared.train["timestamp"].quantile(
        1.0 - validation_fraction, interpolation="nearest"
    )

    for user_index, row in enumerate(grouped.iter_rows(named=True), start=1):
        items = list(row["item_id"])
        timestamps = list(row["timestamp"])
        sequence_index = len(sequences)
        sequences.append((items, timestamps))
        for position in range(min_history_len, len(items)):
            timestamp = timestamps[position]
            if timestamp >= prepared.test_start_timestamp:
                destination = test_examples
                limit = max_eval_examples
            elif timestamp >= validation_start:
                destination = validation_examples
                limit = max_eval_examples
            else:
                destination = train_examples
                limit = max_train_examples
            if limit is not None and len(destination) >= limit:
                continue

            positive = items[position]
            history = items[:position]
            candidates = [(positive, 1.0)]
            candidates.extend(
                (
                    _negative_item(rng, prepared.catalog_size, positive, history),
                    0.0,
                )
                for _ in range(negatives_per_positive)
            )
            for candidate, target in candidates:
                if limit is not None and len(destination) >= limit:
                    break
                destination.append(
                    RankingExample(
                        sequence_index=sequence_index,
                        position=position,
                        user_index=user_index,
                        candidate_item=candidate,
                        timestamp=timestamp,
                        target=target,
                    )
                )

    for name, examples in (
        ("train", train_examples),
        ("validation", validation_examples),
        ("test", test_examples),
    ):
        if not examples:
            raise ValueError(f"ranking {name} examples are empty")
    return RankingData(
        sequences=sequences,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        sid_codes=sid_codes,
        sid_cardinalities=sid_cardinalities,
        num_users=grouped.height + 1,
    )


def build_ranking_loaders(
    data: RankingData,
    *,
    max_history_len: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_semantic_ranking,
    }
    datasets = [
        SemanticRankingDataset(
            data.sequences, examples, data.sid_codes, max_history_len
        )
        for examples in (
            data.train_examples,
            data.validation_examples,
            data.test_examples,
        )
    ]
    return (
        DataLoader(datasets[0], shuffle=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )
