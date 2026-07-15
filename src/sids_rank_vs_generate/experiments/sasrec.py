from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..data import PreparedInteractions, build_histories, build_targets
from ..evaluation import evaluate_ranking
from ..pipeline import PipelineConfig, run_pipeline


@dataclass(frozen=True)
class SASRecConfig:
    embedding_dim: int = 64
    max_seq_len: int = 100
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    topk: int = 100
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_ints = {
            "embedding_dim": self.embedding_dim,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "topk": self.topk,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class SASRecResult:
    metrics: dict[str, float]
    train_losses: list[float]
    device: str
    train_examples: int
    eval_users: int

    def as_dict(
        self,
        prepared: PreparedInteractions,
        config: SASRecConfig,
    ) -> dict[str, Any]:
        return {
            "experiment": "sasrec_baseline",
            "data": prepared.summary(),
            "config": asdict(config),
            "device": self.device,
            "train_examples": self.train_examples,
            "eval_users": self.eval_users,
            "train_losses": self.train_losses,
            "metrics": self.metrics,
        }


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        histories: dict[Any, list[int]],
        max_seq_len: int,
    ) -> None:
        self.samples: list[tuple[list[int], int]] = []
        for items in histories.values():
            for target_position in range(1, len(items)):
                start = max(0, target_position - max_seq_len)
                self.samples.append(
                    (list(items[start:target_position]), items[target_position])
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.samples[index]


class SASRecEvalDataset(Dataset):
    def __init__(
        self,
        histories: dict[Any, list[int]],
        targets: dict[Any, list[int]],
        max_seq_len: int,
    ) -> None:
        self.targets = {
            uid: list(targets[uid])
            for uid in targets
            if uid in histories and histories[uid]
        }
        self.samples = [
            (uid, list(histories[uid][-max_seq_len:]))
            for uid in self.targets
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, list[int]]:
        return self.samples[index]


def _collate_train(
    batch: list[tuple[list[int], int]],
    padding_item_id: int,
) -> dict[str, torch.Tensor]:
    histories, targets = zip(*batch)
    tensors = [torch.tensor(history, dtype=torch.long) for history in histories]
    return {
        "history": pad_sequence(
            tensors,
            batch_first=True,
            padding_value=padding_item_id,
        ),
        "length": torch.tensor([len(history) for history in histories]),
        "target": torch.tensor(targets, dtype=torch.long),
    }


def _collate_eval(
    batch: list[tuple[Any, list[int]]],
    padding_item_id: int,
) -> dict[str, Any]:
    uids, histories = zip(*batch)
    tensors = [torch.tensor(history, dtype=torch.long) for history in histories]
    return {
        "uid": list(uids),
        "history": pad_sequence(
            tensors,
            batch_first=True,
            padding_value=padding_item_id,
        ),
        "length": torch.tensor([len(history) for history in histories]),
    }


class SASRecModel(nn.Module):
    def __init__(self, num_items: int, config: SASRecConfig) -> None:
        super().__init__()
        self.num_items = num_items
        self.padding_item_id = num_items
        self.max_seq_len = config.max_seq_len
        self.item_embeddings = nn.Embedding(
            num_items + 1,
            config.embedding_dim,
            padding_idx=self.padding_item_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_len,
            config.embedding_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embedding_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.embedding_dim),
            enable_nested_tensor=False,
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.item_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)
        self.item_embeddings.weight[self.padding_item_id].zero_()

    def encode(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if history.shape[1] > self.max_seq_len:
            raise ValueError("history is longer than max_seq_len")
        positions = torch.arange(history.shape[1], device=history.device)
        key_padding_mask = history.eq(self.padding_item_id)
        causal_mask = torch.ones(
            history.shape[1],
            history.shape[1],
            dtype=torch.bool,
            device=history.device,
        ).triu(diagonal=1)
        encoded = self.encoder(
            self.input_dropout(
                self.item_embeddings(history)
                + self.position_embeddings(positions).unsqueeze(0)
            ),
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        last_positions = lengths.to(history.device) - 1
        batch_indices = torch.arange(history.shape[0], device=history.device)
        return encoded[batch_indices, last_positions]

    def forward(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode(history, lengths)
        return user_vectors @ self.item_embeddings.weight[: self.num_items].T


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _move_tensor_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def run_sasrec(
    prepared: PreparedInteractions,
    config: SASRecConfig | None = None,
) -> tuple[SASRecModel, SASRecResult]:
    config = config or SASRecConfig()
    if prepared.train.is_empty() or prepared.test.is_empty():
        raise ValueError("SASRec requires non-empty train and test splits")

    seed_everything(config.seed)
    device = _resolve_device(config.device)
    histories = build_histories(prepared.train)
    targets = build_targets(prepared.test)
    train_dataset = SASRecTrainDataset(histories, config.max_seq_len)
    eval_dataset = SASRecEvalDataset(histories, targets, config.max_seq_len)
    if not train_dataset:
        raise ValueError("SASRec requires at least one next-item train example")
    if not eval_dataset:
        raise ValueError("SASRec requires eval users with non-empty train histories")

    padding_item_id = prepared.catalog_size
    generator = torch.Generator().manual_seed(config.seed)
    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        collate_fn=partial(_collate_train, padding_item_id=padding_item_id),
        **common_loader_args,
    )
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=partial(_collate_eval, padding_item_id=padding_item_id),
        **common_loader_args,
    )

    model = SASRecModel(prepared.catalog_size, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    train_losses: list[float] = []
    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        example_count = 0
        for batch in train_loader:
            batch = _move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch["history"], batch["length"])
                loss = F.cross_entropy(logits, batch["target"])
            loss.backward()
            optimizer.step()
            batch_size = batch["target"].shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            example_count += batch_size
        train_losses.append(total_loss / example_count)

    model.eval()
    candidates: dict[Any, list[int]] = {}
    effective_topk = min(config.topk, prepared.catalog_size)
    with torch.inference_mode():
        for batch in eval_loader:
            histories_tensor = batch["history"].to(device, non_blocking=True)
            lengths_tensor = batch["length"].to(device, non_blocking=True)
            scores = model(histories_tensor, lengths_tensor)
            recommendations = torch.topk(
                scores,
                k=effective_topk,
                dim=1,
            ).indices.cpu().tolist()
            candidates.update(zip(batch["uid"], recommendations))

    eval_targets = eval_dataset.targets
    metrics = evaluate_ranking(
        targets=eval_targets,
        candidates=candidates,
        catalog_size=prepared.catalog_size,
        topk=config.topk,
    )
    if not all(math.isfinite(value) for value in train_losses):
        raise RuntimeError("non-finite SASRec training loss")
    result = SASRecResult(
        metrics=metrics,
        train_losses=train_losses,
        device=str(device),
        train_examples=len(train_dataset),
        eval_users=len(eval_dataset),
    )
    return model, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate SASRec baseline.")
    parser.add_argument("--interactions", type=Path, default=Path("interactions.parquet"))
    parser.add_argument("--core-min-interactions", type=int, default=5)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=args.interactions,
            core_min_interactions_per_item=args.core_min_interactions,
            test_days=args.test_days,
            seed=args.seed,
        )
    )
    config = SASRecConfig(
        embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        topk=args.topk,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    _, result = run_sasrec(prepared, config)
    payload = result.as_dict(prepared, config)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(rendered + "\n", encoding="utf-8")
    return 0
