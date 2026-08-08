from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

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
from ..sids.transformer_user_sid_mapper import (
    TransformerUserSidMapper,
    UserSidMapperConfig,
)
from ..sids.user_aware import user_sid_columns
from ..sids.user_sid_mapper import FrequencyUserSidMapper, IUserSidMapper
from ..tracking import flatten_metrics, flatten_parameters
from ..training.semantic_trainer import train_semantic
from .beam_search import SemanticIDCatalog
from .personalized_generation import validate_personalized_generation
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
    constrain_to_catalog: bool = True
    resolve_collisions: bool = True
    max_train_examples: int | None = None
    max_validation_examples: int | None = 50_000
    max_test_examples: int | None = 50_000
    mapper_top_k: int = 3
    mapper_beam_chunk_size: int = 16
    mapper: UserSidMapperConfig = UserSidMapperConfig()
    use_bf16: bool = True
    device: str = "auto"
    checkpoint_path: Path = Path("best_model.pt")
    num_levels: int = 0
    codebook_size: int = 0
    codebook_sizes: tuple[int, ...] | None = None
    rq_num_levels: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.min_history_len <= 0 or self.max_history_len < self.min_history_len:
            raise ValueError("invalid generation history lengths")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.batch_size_val <= 0:
            raise ValueError("generation epochs and batch sizes must be positive")
        if self.mapper_top_k <= 0 or self.mapper_beam_chunk_size <= 0:
            raise ValueError("mapper top-k and beam chunk size must be positive")
        if self.codebook_sizes is not None and any(
            size <= 0 for size in self.codebook_sizes
        ):
            raise ValueError("codebook_sizes must be positive")


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


def _sid_sizes(frame: pl.DataFrame, columns: list[str]) -> tuple[int, ...]:
    sizes = []
    for column in columns:
        maximum = frame[column].max()
        if maximum is None:
            raise ValueError(f"SID column {column} must not be empty")
        sizes.append(int(cast(Any, maximum)) + 1)
    return tuple(sizes)


def _user_sid_sizes(pipeline, frame: pl.DataFrame) -> tuple[int, ...]:
    columns = user_sid_columns(frame)
    config = getattr(pipeline, "config", None)
    configured = getattr(config, "codebook_sizes", None)
    if configured is None:
        size = getattr(config, "codebook_size", None)
        levels = getattr(config, "num_levels", None)
        if size is not None and levels is not None:
            configured = (int(size),) * int(levels)
    if configured is not None and len(configured) == len(columns):
        return tuple(int(size) for size in configured)
    return _sid_sizes(frame, columns)


def _build_sid_catalog(
    frame: pl.DataFrame,
    columns: list[str],
    codebook_sizes: tuple[int, ...],
) -> SemanticIDCatalog:
    return SemanticIDCatalog(
        semantic_ids=frame.select(*columns).iter_rows(),
        codebook_sizes=codebook_sizes,
    )


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
        base_codebook_size = max(_sid_sizes(base_semantic_df, base_sid_columns))
        max_collision = int(
            cast(
                Any,
                base_semantic_df.group_by(base_sid_columns).len()["len"].max(),
            )
        )
        collision_codebook_size = max(base_codebook_size, max_collision)
        semantic_df = (
            add_collision_resolving_sid(
                base_semantic_df,
                codebook_size=collision_codebook_size,
            )
            if self.config.resolve_collisions
            else base_semantic_df
        )
        global_sid_columns = get_sid_cols(semantic_df)
        global_sid_sizes = _sid_sizes(semantic_df, global_sid_columns)

        pair_semantic_df = None
        target_frame = semantic_df
        target_sid_columns = global_sid_columns
        target_sid_sizes = global_sid_sizes
        if context.user_sid_pipeline is not None:
            all_pairs = (
                context.prepared.data.select("uid", "item_id")
                .unique()
                .sort(["uid", "item_id"])
            )
            pair_semantic_df = context.user_sid_pipeline.update(all_pairs)
            target_frame = pair_semantic_df
            target_sid_columns = user_sid_columns(pair_semantic_df)
            target_sid_sizes = _user_sid_sizes(
                context.user_sid_pipeline,
                pair_semantic_df,
            )

        runtime = replace(
            self.config,
            num_levels=len(target_sid_columns),
            rq_num_levels=len(base_sid_columns),
            codebook_size=max(target_sid_sizes),
            codebook_sizes=target_sid_sizes,
            device=_resolve_device(self.config.device),
            checkpoint_path=context.artifact_dir / "best_model.pt",
        )
        train_examples, validation_examples, test_examples, item2idx = (
            prepare_generation_examples(
                context.prepared,
                semantic_df,
                runtime,
                pair_semantic_df=pair_semantic_df,
            )
        )
        train_examples = limit_examples(
            train_examples,
            runtime.max_train_examples,
            SID_EXPERIMENT_SEED,
        )
        validation_examples = limit_examples(
            validation_examples,
            runtime.max_validation_examples,
            SID_EXPERIMENT_SEED,
        )
        test_examples = limit_examples(
            test_examples,
            runtime.max_test_examples,
            SID_EXPERIMENT_SEED,
        )
        if not train_examples or not validation_examples or not test_examples:
            raise ValueError(
                "generation train, validation, and test examples must be non-empty"
            )

        train_loader, validation_loader, test_loader = build_generation_loaders(
            train_examples,
            validation_examples,
            test_examples,
            runtime,
        )
        sid_to_items = _build_sid_to_items(semantic_df, item2idx)
        sid_catalog = (
            _build_sid_catalog(target_frame, target_sid_columns, target_sid_sizes)
            if runtime.constrain_to_catalog
            else None
        )
        model = SemanticTransformerModel(len(item2idx), runtime).to(runtime.device)

        mapper_metrics = None
        with context.tracker.start_run(
            "generation",
            nested=True,
            tags={"stage": "generation", "model": "semantic_transformer"},
        ) as tracker:
            tracker.log_params(
                flatten_parameters(
                    {
                        "config": asdict(runtime),
                        "target_sid_type": (
                            "personalized" if pair_semantic_df is not None else "global"
                        ),
                        "examples": {
                            "train": len(train_examples),
                            "validation": len(validation_examples),
                            "test": len(test_examples),
                        },
                    }
                )
            )
            if pair_semantic_df is not None:
                mapper, mapper_metrics = self._fit_mapper(
                    context=context,
                    runtime=runtime,
                    semantic_df=semantic_df,
                    global_sid_columns=global_sid_columns,
                    global_sid_sizes=global_sid_sizes,
                    user_sid_sizes=target_sid_sizes,
                    sid_to_items=sid_to_items,
                    validation_examples=validation_examples,
                    test_examples=test_examples,
                )

                def personalized_validation(model, val_loader, cfg):
                    return validate_personalized_generation(
                        model,
                        val_loader,
                        cfg,
                        mapper,
                        sid_to_items,
                        sid_catalog=sid_catalog,
                    )

                train_semantic(
                    model,
                    train_loader,
                    validation_loader,
                    runtime,
                    validation_callback=personalized_validation,
                    tracker=tracker,
                )
                validation_metrics = personalized_validation(
                    model,
                    validation_loader,
                    runtime,
                )
                metrics = personalized_validation(model, test_loader, runtime)
            else:
                train_semantic(
                    model,
                    train_loader,
                    validation_loader,
                    runtime,
                    sid_to_items=sid_to_items,
                    sid_catalog=sid_catalog,
                    tracker=tracker,
                )
                validation_metrics = self._validate_global(
                    model,
                    validation_loader,
                    runtime,
                    sid_to_items,
                    sid_catalog,
                )
                metrics = self._validate_global(
                    model,
                    test_loader,
                    runtime,
                    sid_to_items,
                    sid_catalog,
                )
            tracker.log_metrics(
                flatten_metrics(
                    {
                        "final.validation": validation_metrics,
                        "final.test": metrics,
                    }
                )
            )

        semantic_df.write_parquet(context.artifact_dir / "semantic_ids.parquet")
        if pair_semantic_df is not None:
            pair_semantic_df.write_parquet(
                context.artifact_dir / "user_semantic_ids_observed.parquet"
            )
        return {
            "config": asdict(runtime),
            "target_sid_type": (
                "personalized" if pair_semantic_df is not None else "global"
            ),
            "examples": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
            },
            "validation_metrics": validation_metrics,
            "metrics": metrics,
            "mapper": mapper_metrics,
            "checkpoint": runtime.checkpoint_path,
        }

    @staticmethod
    def _validate_global(
        model,
        loader,
        runtime: GenerationConfig,
        sid_to_items,
        sid_catalog,
    ) -> dict[str, Any]:
        if runtime.resolve_collisions:
            return validate_semantic_model_without_collisions(
                model,
                loader,
                runtime,
                sid_catalog=sid_catalog,
                sid_to_items=sid_to_items,
            )
        return validate_semantic_model_with_collisions(
            model,
            loader,
            runtime,
            sid_to_items,
            sid_catalog=sid_catalog,
        )

    @staticmethod
    def _fit_mapper(
        *,
        context: EvaluationContext,
        runtime: GenerationConfig,
        semantic_df: pl.DataFrame,
        global_sid_columns: list[str],
        global_sid_sizes: tuple[int, ...],
        user_sid_sizes: tuple[int, ...],
        sid_to_items: dict[tuple[int, ...], list[int]],
        validation_examples: list[dict],
        test_examples: list[dict],
    ) -> tuple[IUserSidMapper, dict[str, Any]]:
        if context.fit_user_sids is None or context.user_sid_pipeline is None:
            raise ValueError("user-aware generation requires fitted user SIDs")
        mapper_training_frame = context.fit_user_sids.join(
            semantic_df.select("item_id", *global_sid_columns),
            on="item_id",
            how="inner",
        )
        checkpoint = None
        with context.tracker.start_run(
            "generation.mapper",
            nested=True,
            tags={"stage": "generation.mapper", "model": runtime.mapper.kind},
        ) as tracker:
            tracker.log_params(
                flatten_parameters(
                    {
                        "config": asdict(runtime.mapper),
                        "train_examples": mapper_training_frame.height,
                        "user_sid_sizes": user_sid_sizes,
                        "global_sid_sizes": global_sid_sizes,
                    }
                )
            )
            training: dict[str, Any]
            if runtime.mapper.kind == "frequency":
                frequency_mapper = FrequencyUserSidMapper()
                frequency_mapper.fit(mapper_training_frame)
                mapper: IUserSidMapper = frequency_mapper
                training = {"train_examples": mapper_training_frame.height}
            else:
                all_users = context.prepared.data.select("uid").unique().sort("uid")
                user_features = context.user_sid_pipeline.user_features(all_users)
                checkpoint = context.artifact_dir / "best_mapper.pt"
                transformer_mapper = TransformerUserSidMapper(
                    runtime.mapper,
                    user_sid_sizes=user_sid_sizes,
                    global_sid_sizes=global_sid_sizes,
                    valid_global_sids=sid_to_items,
                    user_features=user_features,
                    device=runtime.device,
                    checkpoint_path=checkpoint,
                )
                if runtime.mapper.load_checkpoint_path is None:
                    transformer_mapper.fit(
                        mapper_training_frame,
                        validation_examples,
                        tracker=tracker,
                    )
                else:
                    transformer_mapper.load(
                        mapper_training_frame,
                        runtime.mapper.load_checkpoint_path,
                    )
                mapper = transformer_mapper
                training = transformer_mapper.training_summary
            validation = mapper.evaluate_examples(
                validation_examples,
                k=runtime.mapper_top_k,
            )
            test = mapper.evaluate_examples(test_examples, k=runtime.mapper_top_k)
            tracker.log_metrics(
                flatten_metrics({"final.validation": validation, "final.test": test})
            )
        return mapper, {
            "type": runtime.mapper.kind,
            "training": training,
            "validation": validation,
            "test": test,
            "checkpoint": checkpoint,
        }
