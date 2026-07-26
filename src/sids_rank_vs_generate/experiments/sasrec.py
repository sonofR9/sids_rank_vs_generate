from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..data import PreparedInteractions, build_histories, build_targets
from ..datasets import resolve_dataset
from ..evaluation import evaluate_ranking
from ..pipeline import PipelineConfig, run_pipeline
from ..reproducibility import seed_everything


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
    training_negatives: int = 100
    eval_item_chunk_size: int = 20_000
    freeze_item_embeddings: bool = False
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
            "training_negatives": self.training_negatives,
            "eval_item_chunk_size": self.eval_item_chunk_size,
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
    content_initialized: bool

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
            "content_initialized": self.content_initialized,
            "train_losses": self.train_losses,
            "metrics": self.metrics,
        }


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        histories: dict[Any, list[int]],
        max_seq_len: int,
    ) -> None:
        self.histories: list[list[int]] = []
        self.sample_offsets: list[int] = []
        sample_count = 0
        for items in histories.values():
            if len(items) < 2:
                continue
            self.histories.append(list(items))
            sample_count += len(items) - 1
            self.sample_offsets.append(sample_count)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.sample_offsets[-1] if self.sample_offsets else 0

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        history_index = bisect_right(self.sample_offsets, index)
        previous_offset = (
            self.sample_offsets[history_index - 1] if history_index else 0
        )
        target_position = index - previous_offset + 1
        items = self.histories[history_index]
        start = max(0, target_position - self.max_seq_len)
        return list(items[start:target_position]), items[target_position]


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
    def __init__(
        self,
        num_items: int,
        config: SASRecConfig,
        initial_item_embeddings: torch.Tensor | None = None,
    ) -> None:
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
        if initial_item_embeddings is not None:
            self.initialize_item_embeddings(
                initial_item_embeddings,
                freeze=config.freeze_item_embeddings,
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.item_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)
        self.item_embeddings.weight[self.padding_item_id].zero_()

    @torch.no_grad()
    def initialize_item_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        freeze: bool,
    ) -> None:
        expected_shape = (self.num_items, self.item_embeddings.embedding_dim)
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                "initial item embeddings have shape "
                f"{tuple(embeddings.shape)}; expected {expected_shape}"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("initial item embeddings contain non-finite values")
        self.item_embeddings.weight[: self.num_items].copy_(
            embeddings.to(
                device=self.item_embeddings.weight.device,
                dtype=self.item_embeddings.weight.dtype,
            )
        )
        self.item_embeddings.weight[self.padding_item_id].zero_()
        self.item_embeddings.weight.requires_grad_(not freeze)

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

    def sampled_logits(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        negative_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode(history, lengths)
        positive_vectors = self.item_embeddings(targets)
        negative_vectors = self.item_embeddings(negative_item_ids)
        positive_scores = (user_vectors * positive_vectors).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum(
            "bd,bnd->bn",
            user_vectors,
            negative_vectors,
        )
        return torch.cat((positive_scores, negative_scores), dim=1)

    def recommend_topk(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
        *,
        topk: int,
        item_chunk_size: int,
    ) -> torch.Tensor:
        """Return exact top-k items without materializing all catalog scores."""

        user_vectors = self.encode(history, lengths)
        effective_topk = min(topk, self.num_items)
        best_scores = user_vectors.new_empty((user_vectors.shape[0], 0))
        best_items = torch.empty(
            (user_vectors.shape[0], 0),
            dtype=torch.long,
            device=user_vectors.device,
        )
        for start in range(0, self.num_items, item_chunk_size):
            end = min(start + item_chunk_size, self.num_items)
            scores = (
                user_vectors
                @ self.item_embeddings.weight[start:end].T
            )
            item_ids = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=user_vectors.device,
            ).expand(user_vectors.shape[0], -1)
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_items = torch.cat((best_items, item_ids), dim=1)
            keep = min(effective_topk, merged_scores.shape[1])
            best_scores, indices = torch.topk(merged_scores, k=keep, dim=1)
            best_items = torch.gather(merged_items, 1, indices)
        return best_items


def load_item_embedding_catalog(path: str | Path) -> pl.DataFrame:
    embeddings_path = Path(path).expanduser()
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"item embeddings parquet not found: {embeddings_path}"
        )
    schema = pl.scan_parquet(embeddings_path).collect_schema()
    required = {"item_id", "normalized_embed"}
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(
            "item embeddings are missing columns: "
            f"{', '.join(missing)}"
        )
    catalog = pl.read_parquet(embeddings_path, columns=["item_id"])
    if catalog.is_empty():
        raise ValueError("item embeddings must not be empty")
    if catalog["item_id"].null_count():
        raise ValueError("item embeddings contain null item_id values")
    if catalog["item_id"].n_unique() != catalog.height:
        raise ValueError("item embeddings must contain one row per item_id")
    return catalog


def load_mapped_item_embeddings(
    path: str | Path,
    prepared: PreparedInteractions,
) -> torch.Tensor:
    """Load normalized content vectors in the shared dense item-id order."""

    embeddings_path = Path(path).expanduser()
    query = pl.scan_parquet(embeddings_path).select(
        "item_id",
        "normalized_embed",
    )
    first = query.select("normalized_embed").head(1).collect()
    if first.is_empty() or first["normalized_embed"][0] is None:
        raise ValueError("normalized_embed must contain non-null vectors")
    width = len(first["normalized_embed"][0])
    if width <= 0:
        raise ValueError("normalized_embed vectors must not be empty")

    mapping = prepared.item_mapping.lazy().rename({"item_id": "mapped_item_id"})
    mapped = (
        query.join(
            mapping,
            left_on="item_id",
            right_on="original_item_id",
            how="inner",
        )
        .select("mapped_item_id", "normalized_embed")
        .sort("mapped_item_id")
        .collect()
    )
    if mapped.height != prepared.catalog_size:
        missing = prepared.catalog_size - mapped.height
        raise ValueError(
            f"normalized embeddings are missing {missing} prepared items"
        )
    if mapped["normalized_embed"].null_count():
        raise ValueError("normalized_embed contains null vectors")
    expected_ids = pl.Series(
        "mapped_item_id",
        range(prepared.catalog_size),
        dtype=pl.Int64,
    )
    if not mapped["mapped_item_id"].equals(expected_ids):
        raise ValueError("normalized embeddings do not match the dense item mapping")

    vector_series = mapped["normalized_embed"]
    if isinstance(vector_series.dtype, pl.Array):
        vectors = vector_series.cast(pl.Array(pl.Float32, width)).to_numpy()
    else:
        vectors = (
            vector_series
            .list.to_array(width)
            .cast(pl.Array(pl.Float32, width))
            .to_numpy()
        )
    matrix = np.array(vectors, dtype=np.float32, copy=True)
    return torch.from_numpy(matrix)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _cuda_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.bfloat16
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def _move_tensor_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _sample_negative_item_ids(
    targets: torch.Tensor,
    *,
    num_items: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_items < 2:
        raise ValueError("negative sampling requires at least two catalog items")
    sampled = torch.randint(
        0,
        num_items - 1,
        (targets.shape[0], count),
        generator=generator,
    )
    targets = targets.cpu().unsqueeze(1)
    return sampled + sampled.ge(targets)


def run_sasrec(
    prepared: PreparedInteractions,
    config: SASRecConfig | None = None,
    *,
    initial_item_embeddings: torch.Tensor | None = None,
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
    negative_generator = torch.Generator().manual_seed(config.seed + 1)
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

    model = SASRecModel(
        prepared.catalog_size,
        config,
        initial_item_embeddings=initial_item_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    autocast_dtype = _cuda_autocast_dtype(device)
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and autocast_dtype == torch.float16,
    )
    train_losses: list[float] = []
    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        example_count = 0
        for batch in train_loader:
            batch["negative_item_id"] = _sample_negative_item_ids(
                batch["target"],
                num_items=prepared.catalog_size,
                count=config.training_negatives,
                generator=negative_generator,
            )
            batch = _move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                logits = model.sampled_logits(
                    batch["history"],
                    batch["length"],
                    batch["target"],
                    batch["negative_item_id"],
                )
                labels = torch.zeros(
                    logits.shape[0],
                    dtype=torch.long,
                    device=device,
                )
                loss = F.cross_entropy(logits, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            batch_size = batch["target"].shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            example_count += batch_size
        train_losses.append(total_loss / example_count)

    model.eval()
    candidates: dict[Any, list[int]] = {}
    with torch.inference_mode():
        for batch in eval_loader:
            histories_tensor = batch["history"].to(device, non_blocking=True)
            lengths_tensor = batch["length"].to(device, non_blocking=True)
            recommendations = model.recommend_topk(
                histories_tensor,
                lengths_tensor,
                topk=config.topk,
                item_chunk_size=config.eval_item_chunk_size,
            ).cpu().tolist()
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
        content_initialized=initial_item_embeddings is not None,
    )
    return model, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate SASRec on official Yambda-50M likes, "
            "initialized with normalized audio embeddings."
        )
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        help="Override the official flat/50m/likes.parquet path.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Override the prepared Yambda-50M embeddings path.",
    )
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        default=Path("embeddings.parquet"),
        help=(
            "One-time source for preparing the compact 50M embedding cache "
            "(default: embeddings.parquet)."
        ),
    )
    parser.add_argument("--core-min-interactions", type=int, default=5)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument(
        "--timestamp-unit-seconds",
        type=float,
        default=5.0,
        help="Yambda timestamps are binned in five-second units.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Defaults to the content-vector width (128).",
    )
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--training-negatives", type=int, default=100)
    parser.add_argument("--eval-item-chunk-size", type=int, default=20_000)
    parser.add_argument("--freeze-item-embeddings", action="store_true")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.interactions is None) != (args.embeddings is None):
        raise ValueError(
            "--interactions and --embeddings must be provided together"
        )
    if args.interactions is None:
        dataset = resolve_dataset(
            "50m",
            source_embeddings_path=args.source_embeddings,
        )
        interactions_path = dataset.interactions_path
        embeddings_path = dataset.embeddings_path
    else:
        interactions_path = args.interactions.expanduser()
        embeddings_path = args.embeddings.expanduser()

    embedding_catalog = load_item_embedding_catalog(embeddings_path)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=interactions_path,
            core_min_interactions_per_item=args.core_min_interactions,
            test_days=args.test_days,
            timestamp_unit_seconds=args.timestamp_unit_seconds,
            seed=args.seed,
        ),
        eligible_items=embedding_catalog,
    )
    initial_item_embeddings = load_mapped_item_embeddings(
        embeddings_path,
        prepared,
    )
    embedding_dim = args.embedding_dim or initial_item_embeddings.shape[1]
    config = SASRecConfig(
        embedding_dim=embedding_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        training_negatives=args.training_negatives,
        eval_item_chunk_size=args.eval_item_chunk_size,
        freeze_item_embeddings=args.freeze_item_embeddings,
        topk=args.topk,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    _, result = run_sasrec(
        prepared,
        config,
        initial_item_embeddings=initial_item_embeddings,
    )
    payload = result.as_dict(prepared, config)
    payload["dataset"] = {
        "name": "Yambda-50M",
        "event_type": "likes",
        "interactions_path": str(interactions_path),
        "embeddings_path": str(embeddings_path),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(rendered + "\n", encoding="utf-8")
    return 0
