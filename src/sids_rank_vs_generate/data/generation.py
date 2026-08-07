from __future__ import annotations

import random
from functools import partial

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..sids.collision_resolution import get_sid_cols
from ..sids.user_aware import user_sid_columns
from .preprocessing import PreparedInteractions


class GenerationDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def prepare_generation_examples(
    prepared: PreparedInteractions,
    semantic_df: pl.DataFrame,
    config,
    pair_semantic_df: pl.DataFrame | None = None,
):
    global_sid_columns = get_sid_cols(semantic_df)
    target_sid_columns = global_sid_columns
    if pair_semantic_df is not None:
        target_sid_columns = user_sid_columns(pair_semantic_df)
    if len(target_sid_columns) != config.num_levels:
        raise ValueError(
            f"Expected {config.num_levels} SID levels, "
            f"received {len(target_sid_columns)}"
        )
    frame = prepared.data.join(
        semantic_df.select("item_id", *global_sid_columns),
        on="item_id",
        how="inner",
    ).sort(["uid", "timestamp"])
    if pair_semantic_df is not None:
        frame = frame.join(
            pair_semantic_df.select("uid", "item_id", *target_sid_columns),
            on=["uid", "item_id"],
            how="inner",
        ).sort(["uid", "timestamp"])
    item_ids = sorted(frame["item_id"].unique().to_list())
    item2idx = {item_id: index + 1 for index, item_id in enumerate(item_ids)}

    train_examples = []
    validation_examples = []
    test_examples = []
    history = []
    current_uid = None
    for row in frame.iter_rows(named=True):
        if row["uid"] != current_uid:
            current_uid = row["uid"]
            history = []
        item_idx = item2idx[row["item_id"]]
        if len(history) >= config.min_history_len:
            example = {
                "uid": row["uid"],
                "history": history[-config.max_history_len :],
                "target": item_idx,
                "sid": [int(row[column]) for column in target_sid_columns],
                "global_sid": [int(row[column]) for column in global_sid_columns],
            }
            if row["timestamp"] >= prepared.test_start_timestamp:
                test_examples.append(example)
            elif row["timestamp"] >= prepared.validation_start_timestamp:
                validation_examples.append(example)
            else:
                train_examples.append(example)
        history.append(item_idx)
    return train_examples, validation_examples, test_examples, item2idx


def limit_examples(
    examples: list[dict],
    count: int | None,
    seed: int,
) -> list[dict]:
    if count is None or len(examples) <= count:
        return examples
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples)), count))
    return [examples[index] for index in indices]


def collate_generation(
    batch: list[dict],
    *,
    num_levels: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_length = max(len(example["history"]) for example in batch)
    history = torch.zeros(batch_size, max_length, dtype=torch.long)
    positions = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for index, example in enumerate(batch):
        values = torch.tensor(example["history"], dtype=torch.long)
        length = len(values)
        history[index, :length] = values
        positions[index, :length] = torch.arange(length)
        mask[index, :length] = True
    sid = torch.tensor([example["sid"] for example in batch], dtype=torch.long)
    if sid.shape != (batch_size, num_levels):
        raise ValueError(f"Unexpected SID shape: {tuple(sid.shape)}")
    return {
        "uid": [example["uid"] for example in batch],
        "history": history,
        "positions": positions,
        "mask": mask,
        "target": torch.tensor(
            [example["target"] for example in batch], dtype=torch.long
        ),
        "sid": sid,
        "global_sid": torch.tensor(
            [example["global_sid"] for example in batch],
            dtype=torch.long,
        ),
    }


def build_generation_loaders(
    train_examples,
    validation_examples,
    test_examples,
    config,
):
    collator = partial(collate_generation, num_levels=config.num_levels)
    train_loader = DataLoader(
        GenerationDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=str(config.device).startswith("cuda"),
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        GenerationDataset(validation_examples),
        batch_size=config.batch_size_val,
        shuffle=False,
        collate_fn=collator,
        pin_memory=str(config.device).startswith("cuda"),
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        GenerationDataset(test_examples),
        batch_size=config.batch_size_val,
        shuffle=False,
        collate_fn=collator,
        pin_memory=str(config.device).startswith("cuda"),
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader, test_loader
