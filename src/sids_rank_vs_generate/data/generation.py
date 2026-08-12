from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..sids.collision_resolution import get_sid_cols
from ..sids.user_aware import user_sid_columns
from .preprocessing import PreparedInteractions
from .protocols import uses_legacy_heldout


class GenerationDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class SIDLogQTable:
    """Train-only prefix probabilities for autoregressive SID LogQ.

    The empirical SID distribution is factorized as
    ``q(sid) = product_l q(z_l | z_<l)``. Additive smoothing is applied to
    every codebook token so validation/catalog prefixes unseen in train remain
    finite. There is deliberately no tunable alpha: consumers subtract these
    log-probabilities with coefficient one.
    """

    def __init__(
        self,
        examples: Sequence[dict],
        *,
        codebook_sizes: Sequence[int],
        beta: float = 1.0,
    ) -> None:
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        if not self.codebook_sizes or any(
            size <= 0 for size in self.codebook_sizes
        ):
            raise ValueError("codebook_sizes must contain positive sizes")
        if beta <= 0 or not math.isfinite(beta):
            raise ValueError("SID LogQ beta must be finite and positive")
        if not examples:
            raise ValueError("cannot estimate SID LogQ from no train examples")
        self.beta = float(beta)

        prefix_counts: list[dict[tuple[int, ...], Counter[int]]] = [
            {} for _ in self.codebook_sizes
        ]
        unique_sids: set[tuple[int, ...]] = set()
        for example in examples:
            sid = tuple(int(token) for token in example["sid"])
            if len(sid) != len(self.codebook_sizes):
                raise ValueError("SID length does not match codebook sizes")
            unique_sids.add(sid)
            for level, (token, size) in enumerate(zip(sid, self.codebook_sizes)):
                if not 0 <= token < size:
                    raise ValueError(f"SID token at level {level} is out of range")
                prefix = sid[:level]
                counter = prefix_counts[level].setdefault(prefix, Counter())
                counter[token] += 1

        uniform: list[torch.Tensor] = []
        for size in self.codebook_sizes:
            uniform.append(
                torch.full((size,), -math.log(size), dtype=torch.float32)
            )
        # Keep counts sparse. Materializing a dense C-vector for every deep
        # prefix can consume hundreds of MB for C=1024; batches cache only the
        # prefixes they actually use.
        self._prefix_counts = tuple(prefix_counts)
        self._uniform = tuple(uniform)
        self.num_train_interactions = len(examples)
        self.num_unique_sids = len(unique_sids)

    def log_q_for_prefix(
        self,
        prefix: Sequence[int],
        *,
        level: int,
    ) -> torch.Tensor:
        if not 0 <= level < len(self.codebook_sizes):
            raise ValueError("SID LogQ level is out of range")
        key = tuple(int(token) for token in prefix)
        if len(key) != level:
            raise ValueError("SID LogQ prefix length must equal level")
        counts = self._prefix_counts[level].get(key)
        if counts is None:
            return self._uniform[level]
        smoothed = torch.full(
            (self.codebook_sizes[level],),
            self.beta,
            dtype=torch.float64,
        )
        for token, count in counts.items():
            smoothed[token] += count
        return (smoothed / smoothed.sum()).log().float()

    def log_q_for_sid(self, sid: Sequence[int]) -> tuple[torch.Tensor, ...]:
        values = tuple(int(token) for token in sid)
        if len(values) != len(self.codebook_sizes):
            raise ValueError("SID length does not match SID LogQ levels")
        return tuple(
            self.log_q_for_prefix(values[:level], level=level)
            for level in range(len(values))
        )


@dataclass(frozen=True)
class UserProfileTable:
    user_to_index: dict[Any, int]
    values: torch.Tensor

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> "UserProfileTable":
        required = {"uid", "user_feature"}
        if not required.issubset(frame.columns):
            raise ValueError("user features must contain uid and user_feature")
        if frame.is_empty() or frame["uid"].n_unique() != frame.height:
            raise ValueError("user features must contain one row per uid")
        users = frame["uid"].to_list()
        values = torch.tensor(
            frame["user_feature"].to_list(),
            dtype=torch.float32,
        )
        if values.ndim != 2 or values.size(1) == 0:
            raise ValueError("user features must form a non-empty matrix")
        if not torch.isfinite(values).all():
            raise ValueError("user features must contain only finite values")
        return cls(
            user_to_index={uid: index for index, uid in enumerate(users)},
            values=values,
        )

    @property
    def dimension(self) -> int:
        return int(self.values.size(1))

    def lookup(self, users: list[Any]) -> torch.Tensor:
        try:
            indices = [self.user_to_index[uid] for uid in users]
        except KeyError as error:
            raise ValueError("user features do not cover the whole batch") from error
        return self.values[indices]


def attach_item_log_q(
    examples: list[dict],
    *,
    eps: float = 1e-12,
) -> list[dict]:
    """Attach log item frequencies for the actual uniformly sampled examples."""

    if not examples:
        raise ValueError("cannot estimate item sampling probabilities from no examples")
    if eps <= 0 or eps > 1:
        raise ValueError("logq eps must be in (0, 1]")
    counts = Counter(int(example["target"]) for example in examples)
    total = len(examples)
    log_q_by_item = {
        item: math.log(max(count / total, eps))
        for item, count in counts.items()
    }
    return [
        {**example, "log_q": log_q_by_item[int(example["target"])]}
        for example in examples
    ]


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
            if (
                uses_legacy_heldout(prepared.protocol_id)
                and row["timestamp"] >= prepared.test_start_timestamp
            ):
                validation_examples.append(example)
                test_examples.append(example)
            elif row["timestamp"] >= prepared.test_start_timestamp:
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
    sid_logq_table: SIDLogQTable | None = None,
    user_profiles: UserProfileTable | None = None,
) -> dict[str, Any]:
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
    result = {
        "history": history,
        "positions": positions,
        "mask": mask,
        "target": torch.tensor(
            [example["target"] for example in batch], dtype=torch.long
        ),
        "sid": sid,
    }
    has_uid = ["uid" in example for example in batch]
    if any(has_uid) and not all(has_uid):
        raise ValueError("either every example or no example must provide uid")
    if all(has_uid):
        result["uid"] = [example["uid"] for example in batch]
    has_global_sid = ["global_sid" in example for example in batch]
    if any(has_global_sid) and not all(has_global_sid):
        raise ValueError(
            "either every example or no example must provide global_sid"
        )
    if all(has_global_sid):
        result["global_sid"] = torch.tensor(
            [example["global_sid"] for example in batch],
            dtype=torch.long,
        )
    has_log_q = ["log_q" in example for example in batch]
    if any(has_log_q) and not all(has_log_q):
        raise ValueError("either every example or no example must provide log_q")
    if all(has_log_q):
        result["log_q"] = torch.tensor(
            [example["log_q"] for example in batch], dtype=torch.float32
        )
    if sid_logq_table is not None:
        sid_values = [
            tuple(int(token) for token in example["sid"])
            for example in batch
        ]
        level_rows = []
        for level in range(num_levels):
            batch_cache: dict[tuple[int, ...], torch.Tensor] = {}
            rows = []
            for sid_value in sid_values:
                prefix = sid_value[:level]
                if prefix not in batch_cache:
                    batch_cache[prefix] = sid_logq_table.log_q_for_prefix(
                        prefix,
                        level=level,
                    )
                rows.append(batch_cache[prefix])
            level_rows.append(torch.stack(rows))
        result["sid_log_q"] = tuple(level_rows)
    if user_profiles is not None:
        if "uid" not in result:
            raise ValueError("uid is required when user profiles are enabled")
        result["user_features"] = user_profiles.lookup(result["uid"])
    return result


def build_generation_loaders(
    train_examples,
    validation_examples,
    test_examples,
    config,
    user_profiles: UserProfileTable | None = None,
    *,
    sid_logq_table: SIDLogQTable | None = None,
):
    collator = partial(
        collate_generation,
        num_levels=config.num_levels,
        sid_logq_table=sid_logq_table,
        user_profiles=user_profiles,
    )
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
