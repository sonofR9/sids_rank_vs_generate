from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl
import torch

from ..data.generation import (
    build_generation_loaders,
    limit_examples,
    prepare_generation_examples,
)
from ..models.semantic_transformer_model import SemanticTransformerModel
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from ..sids.collision_resolution import add_collision_resolving_sid, get_sid_cols
from ..training.semantic_trainer import train_semantic
from .semantic_validation_with_collision import (
    validate_semantic_model_with_collisions,
)
from .semantic_validation_without_collision import (
    validate_semantic_model_without_collisions,
)
from .stages import EvaluationContext, IEvaluationStage


@dataclass(frozen=True)
class GenerationConfig:
    max_history_len: int = 16
    min_history_len: int = 5
    d_model: int = 128
    d_ff: int = 512
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 1024
    batch_size_val: int = 64
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 30
    grad_clip: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    num_beams: int = 150
    recall_k: int = 100
    resolve_collisions: bool = True
    max_train_examples: int | None = None
    max_validation_examples: int | None = 50_000
    use_bf16: bool = True
    device: str = "auto"
    checkpoint_path: Path = Path("best_model.pt")
    num_levels: int = 0
    codebook_size: int = 0
    rq_num_levels: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.min_history_len <= 0 or self.max_history_len < self.min_history_len:
            raise ValueError("invalid generation history lengths")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.batch_size_val <= 0:
            raise ValueError("generation epochs and batch sizes must be positive")


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return name


def _build_sid_to_items(
    semantic_df: pl.DataFrame,
    item2idx: dict[int, int],
) -> dict[tuple[int, ...], list[int]]:
    sid_columns = get_sid_cols(semantic_df)
    result: dict[tuple[int, ...], list[int]] = {}
    for row in semantic_df.select(*sid_columns, "item_id").iter_rows(named=True):
        sid = tuple(int(row[column]) for column in sid_columns)
        result.setdefault(sid, []).append(item2idx[row["item_id"]])
    return result


class GenerationStage(IEvaluationStage):
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "generation"

    def evaluate(self, context: EvaluationContext) -> dict[str, Any]:
        seed_everything(SID_EXPERIMENT_SEED)
        base_semantic_df = context.items_with_sids.select(
            "item_id", *get_sid_cols(context.items_with_sids)
        )
        base_sid_columns = get_sid_cols(base_semantic_df)
        base_codebook_size = max(
            int(base_semantic_df[column].max()) + 1 for column in base_sid_columns
        )
        max_collision = int(
            base_semantic_df.group_by(base_sid_columns).len()["len"].max()
        )
        codebook_size = max(base_codebook_size, max_collision)

        if self.config.resolve_collisions:
            semantic_df = add_collision_resolving_sid(
                base_semantic_df, codebook_size=codebook_size
            )
        else:
            semantic_df = base_semantic_df
        sid_columns = get_sid_cols(semantic_df)
        runtime = replace(
            self.config,
            num_levels=len(sid_columns),
            rq_num_levels=len(base_sid_columns),
            codebook_size=codebook_size,
            device=_resolve_device(self.config.device),
            checkpoint_path=context.artifact_dir / "best_model.pt",
        )

        train_examples, validation_examples, item2idx = (
            prepare_generation_examples(
                context.prepared,
                semantic_df,
                runtime,
            )
        )
        train_examples = limit_examples(
            train_examples, runtime.max_train_examples, SID_EXPERIMENT_SEED
        )
        validation_examples = limit_examples(
            validation_examples,
            runtime.max_validation_examples,
            SID_EXPERIMENT_SEED,
        )
        if not train_examples or not validation_examples:
            raise ValueError("generation train and validation examples must be non-empty")
        train_loader, validation_loader = build_generation_loaders(
            train_examples, validation_examples, runtime
        )
        sid_to_items = (
            None
            if runtime.resolve_collisions
            else _build_sid_to_items(semantic_df, item2idx)
        )
        model = SemanticTransformerModel(len(item2idx), runtime).to(runtime.device)
        train_semantic(
            model,
            train_loader,
            validation_loader,
            runtime,
            sid_to_items=sid_to_items,
        )
        if runtime.resolve_collisions:
            metrics = validate_semantic_model_without_collisions(
                model, validation_loader, runtime
            )
        else:
            metrics = validate_semantic_model_with_collisions(
                model, validation_loader, runtime, sid_to_items
            )
        semantic_df.write_parquet(context.artifact_dir / "semantic_ids.parquet")
        return {
            "config": asdict(runtime),
            "examples": {
                "train": len(train_examples),
                "validation": len(validation_examples),
            },
            "metrics": metrics,
            "checkpoint": runtime.checkpoint_path,
        }
