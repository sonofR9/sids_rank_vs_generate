from functools import partial
import random
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


class RecDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def prepare_examples(
    data: pl.DataFrame,
    semantic_df: pl.DataFrame,
    cfg,
):
    sid_cols = get_sid_cols(semantic_df)

    if len(sid_cols) != cfg.num_levels:
        raise ValueError(
            f"Expected {cfg.num_levels} SID levels, "
            f"received {len(sid_cols)}"
        )

    df = (
        data
        .with_columns(
            (
                pl.col("timestamp") // cfg.seconds_per_day
            ).alias("day")
        )
        .join(
            semantic_df,
            on="item_id",
            how="inner",
        )
        .sort(["uid", "timestamp"])
    )

    item_ids = sorted(df["item_id"].unique().to_list())

    item2idx = {
        item_id: index + 1
        for index, item_id in enumerate(item_ids)
    }

    idx2item = {
        index: item_id
        for item_id, index in item2idx.items()
    }

    days = sorted(df["day"].unique().to_list())
    val_days = set(days[-cfg.val_num_days:])

    examples = []
    history = []
    current_uid = None

    for row in df.iter_rows(named=True):
        uid = int(row["uid"])
        item_id = int(row["item_id"])
        day = int(row["day"])

        if uid != current_uid:
            current_uid = uid
            history = []

        item_idx = item2idx[item_id]

        if len(history) >= cfg.min_history_len:
            examples.append(
                {
                    "history": history[-cfg.max_history_len:],
                    "target": item_idx,
                    "sid": [
                        int(row[column])
                        for column in sid_cols
                    ],
                    "is_val": day in val_days,
                }
            )

        history.append(item_idx)

    train_examples = [
        example
        for example in examples
        if not example["is_val"]
    ]

    val_examples = [
        example
        for example in examples
        if example["is_val"]
    ]

    return train_examples, val_examples, item2idx, idx2item


def collate_fn(
    batch: list[dict],
    num_levels: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(len(example["history"]) for example in batch)

    history = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long,
    )

    positions = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long,
    )

    mask = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    target = torch.tensor(
        [example["target"] for example in batch],
        dtype=torch.long,
    )

    sid = torch.tensor(
        [example["sid"] for example in batch],
        dtype=torch.long,
    )

    if sid.shape != (batch_size, num_levels):
        raise ValueError(
            f"Unexpected SID shape: {tuple(sid.shape)}"
        )

    for index, example in enumerate(batch):
        values = torch.tensor(
            example["history"],
            dtype=torch.long,
        )

        length = len(values)

        history[index, :length] = values
        positions[index, :length] = torch.arange(length)
        mask[index, :length] = True

    return {
        "history": history,
        "positions": positions,
        "mask": mask,
        "target": target,
        "sid": sid,
    }


def limit_examples(
    examples: list[dict],
    count: int | None,
    seed: int,
) -> list[dict]:
    if count is None or len(examples) <= count:
        return examples

    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), count)

    return [examples[index] for index in sorted(indices)]


def build_loaders(
    train_examples,
    val_examples,
    cfg,
):
    collator = partial(
        collate_fn,
        num_levels=cfg.num_levels,
    )

    train_loader = DataLoader(
        RecDataset(train_examples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        RecDataset(val_examples),
        batch_size=cfg.batch_size_val,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader