from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import polars as pl
import torch

from ..data.generation import (
    SIDLogQTable,
    UserProfileTable,
    attach_item_log_q,
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
from ..sids.user_sid_mapper import (
    FrequencyUserSidMapper,
    IUserSidMapper,
    mapping_ambiguity_metrics,
    mapping_fit_support_metrics,
)
from ..tracking import flatten_metrics, flatten_parameters
from ..training.semantic_trainer import train_semantic
from .beam_search import SemanticIDCatalog
from .mapper_inference_diagnostic import (
    diagnose_mapper_inference,
    mapper_inference_mlflow_summary,
)
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
    run_diagnostics: bool = False
    run_mapper_inference_diagnostic: bool = False
    use_user_profile: bool = False
    user_profile_dim: int = 0
    initialize_item_embeddings_from_content: bool = False
    content_embedding_column: str = "normalized_embed"
    content_embedding_scale: float = 1.0
    pad_content_embeddings_to_model_dim: bool = False
    sid_only_validation_during_training: bool = False
    validation_metric: str | None = None
    use_bf16: bool = True
    device: str = "auto"
    checkpoint_path: Path = Path("best_model.pt")
    load_checkpoint_path: Path | None = None
    reference_artifact_dir: Path | None = None
    num_levels: int = 0
    codebook_size: int = 0
    codebook_sizes: tuple[int, ...] | None = None
    rq_num_levels: int = 0
    semantic_history_weight: float = 0.0
    auxiliary_item_loss_weight: float = 0.0
    ranking_temperature: float = 0.07
    logq_mode: str = "none"
    logq_eps: float = 1e-12
    logq_debug: bool = False
    gather_distributed_negatives: bool = False
    sid_logq_mode: str = "none"
    sid_logq_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.min_history_len <= 0 or self.max_history_len < self.min_history_len:
            raise ValueError("invalid generation history lengths")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.batch_size_val <= 0:
            raise ValueError("generation epochs and batch sizes must be positive")
        if self.ranking_temperature <= 0:
            raise ValueError("ranking temperature must be positive")
        if self.logq_mode not in {"none", "standard", "corrected"}:
            raise ValueError("logq_mode must be one of: none, standard, corrected")
        if self.logq_eps <= 0 or self.logq_eps > 1:
            raise ValueError("logq_eps must be in (0, 1]")
        if self.sid_logq_mode not in {"none", "standard"}:
            raise ValueError("sid_logq_mode must be one of: none, standard")
        if self.sid_logq_beta <= 0:
            raise ValueError("sid_logq_beta must be positive")
        if self.mapper_top_k <= 0 or self.mapper_beam_chunk_size <= 0:
            raise ValueError("mapper top-k and beam chunk size must be positive")
        if self.user_profile_dim < 0:
            raise ValueError("user_profile_dim must be non-negative")
        if self.auxiliary_item_loss_weight < 0:
            raise ValueError("auxiliary_item_loss_weight must be non-negative")
        if not self.content_embedding_column:
            raise ValueError("content_embedding_column must not be empty")
        if self.content_embedding_scale <= 0:
            raise ValueError("content_embedding_scale must be positive")
        if self.validation_metric is not None and not self.validation_metric:
            raise ValueError("validation_metric must not be empty")
        if self.codebook_sizes is not None and any(
            size <= 0 for size in self.codebook_sizes
        ):
            raise ValueError("codebook_sizes must be positive")
        if self.run_mapper_inference_diagnostic and (
            self.load_checkpoint_path is None
            or self.mapper.kind != "transformer"
            or self.mapper.load_checkpoint_path is None
        ):
            raise ValueError(
                "mapper inference diagnostic requires saved generator and "
                "Transformer mapper checkpoints"
            )


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


def _initial_item_embeddings(
    items: pl.DataFrame,
    item2idx: dict[int, int],
    *,
    column: str,
    d_model: int,
    scale: float = 1.0,
    pad_to_model_dim: bool = False,
) -> torch.Tensor:
    if column not in items.columns:
        raise ValueError(f"item features must contain {column!r}")
    rows = items.select("item_id", column)
    values = torch.tensor(rows[column].to_list(), dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError(
            f"content embeddings must have dimension {d_model}, "
            "received invalid"
        )
    if values.size(1) != d_model:
        if not pad_to_model_dim or values.size(1) > d_model:
            raise ValueError(
                f"content embeddings must have dimension {d_model}, "
                f"received {values.size(1)}"
            )
        padded = torch.zeros(values.size(0), d_model, dtype=values.dtype)
        padded[:, : values.size(1)] = values
        values = padded
    if not torch.isfinite(values).all():
        raise ValueError("content embeddings must contain only finite values")
    values = values * float(scale)
    result = torch.zeros(len(item2idx) + 1, d_model, dtype=torch.float32)
    seen = set()
    for row_index, item_id in enumerate(rows["item_id"].to_list()):
        try:
            mapped_index = item2idx[item_id]
        except KeyError as error:
            raise ValueError("content embeddings contain an unknown item") from error
        result[mapped_index] = values[row_index]
        seen.add(item_id)
    missing = set(item2idx) - seen
    if missing:
        raise ValueError("content embeddings do not cover all generation items")
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

        user_profiles = None
        user_profile_dim = 0
        if self.config.use_user_profile:
            if context.user_sid_pipeline is None:
                raise ValueError(
                    "user profile conditioning requires a user-aware SID pipeline"
                )
            users = context.prepared.data.select("uid").unique().sort("uid")
            user_profiles = UserProfileTable.from_frame(
                context.user_sid_pipeline.user_features(users)
            )
            user_profile_dim = user_profiles.dimension
        if (
            self.config.sid_only_validation_during_training
            and context.user_sid_pipeline is None
        ):
            raise ValueError("SID-only validation requires a user-aware SID pipeline")

        runtime = replace(
            self.config,
            num_levels=len(target_sid_columns),
            rq_num_levels=len(base_sid_columns),
            codebook_size=max(target_sid_sizes),
            codebook_sizes=target_sid_sizes,
            user_profile_dim=user_profile_dim,
            device=_resolve_device(self.config.device),
            checkpoint_path=context.artifact_dir / "best_model.pt",
        )
        if runtime.reference_artifact_dir is not None:
            self._validate_reference_artifacts(
                runtime.reference_artifact_dir,
                semantic_df=semantic_df,
                fit_user_sids=context.fit_user_sids,
                observed_user_sids=pair_semantic_df,
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

        if runtime.logq_mode != "none":
            # The loader samples training examples uniformly, so the in-batch
            # item proposal Q is the target frequency in this final train set.
            train_examples = attach_item_log_q(
                train_examples,
                eps=runtime.logq_eps,
            )
        sid_logq_table = None
        if runtime.sid_logq_mode == "standard":
            sid_logq_table = SIDLogQTable(
                train_examples,
                codebook_sizes=target_sid_sizes,
                beta=runtime.sid_logq_beta,
            )

        train_loader, validation_loader, test_loader = build_generation_loaders(
            train_examples,
            validation_examples,
            test_examples,
            runtime,
            user_profiles=user_profiles,
            sid_logq_table=sid_logq_table,
        )
        sid_to_items = _build_sid_to_items(semantic_df, item2idx)
        sid_catalog = (
            _build_sid_catalog(target_frame, target_sid_columns, target_sid_sizes)
            if runtime.constrain_to_catalog
            else None
        )
        initial_item_embeddings = None
        if runtime.initialize_item_embeddings_from_content:
            initial_item_embeddings = _initial_item_embeddings(
                context.items_with_sids,
                item2idx,
                column=runtime.content_embedding_column,
                d_model=runtime.d_model,
                scale=runtime.content_embedding_scale,
                pad_to_model_dim=(
                    runtime.pad_content_embeddings_to_model_dim
                ),
            )
        model = SemanticTransformerModel(
            len(item2idx),
            runtime,
            initial_item_embeddings=initial_item_embeddings,
        ).to(runtime.device)

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
                        "sid_logq": {
                            "alpha": 1.0 if sid_logq_table is not None else None,
                            "train_interactions": (
                                sid_logq_table.num_train_interactions
                                if sid_logq_table is not None
                                else None
                            ),
                            "unique_sids": (
                                sid_logq_table.num_unique_sids
                                if sid_logq_table is not None
                                else None
                            ),
                        },
                    }
                )
            )
            if pair_semantic_df is not None:
                mapper, mapper_metrics, comparison_mappers = self._fit_mapper(
                    context=context,
                    runtime=runtime,
                    semantic_df=semantic_df,
                    global_sid_columns=global_sid_columns,
                    global_sid_sizes=global_sid_sizes,
                    user_sid_sizes=target_sid_sizes,
                    observed_user_sids=pair_semantic_df,
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
                        comparison_mappers=comparison_mappers,
                    )

                def personalized_sid_validation(model, val_loader, cfg):
                    return validate_personalized_generation(
                        model,
                        val_loader,
                        cfg,
                        sid_catalog=sid_catalog,
                    )

                if runtime.load_checkpoint_path is None:
                    train_semantic(
                        model,
                        train_loader,
                        validation_loader,
                        runtime,
                        sid_logq_table=sid_logq_table,
                        validation_callback=(
                            personalized_sid_validation
                            if runtime.sid_only_validation_during_training
                            else personalized_validation
                        ),
                        tracker=tracker,
                    )
                else:
                    self._load_generator_checkpoint(
                        model,
                        runtime.load_checkpoint_path,
                        runtime.device,
                        runtime,
                    )
                validation_metrics = personalized_validation(
                    model,
                    validation_loader,
                    runtime,
                )
                metrics = personalized_validation(model, test_loader, runtime)
                if runtime.run_mapper_inference_diagnostic:
                    inference_diagnostic = diagnose_mapper_inference(
                        model,
                        test_loader,
                        runtime,
                        mapper,
                        sid_to_items,
                        sid_catalog=sid_catalog,
                    )
                    tracker.log_metrics(
                        flatten_metrics(
                            {
                                "mapper_inference_diagnostic": (
                                    mapper_inference_mlflow_summary(
                                        inference_diagnostic
                                    )
                                )
                            }
                        )
                    )
                    diagnostic_path = (
                        context.artifact_dir / "mapper_inference_diagnostic.json"
                    )
                    diagnostic_path.write_text(
                        json.dumps(
                            inference_diagnostic,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    mapper_metrics["inference_diagnostic"] = inference_diagnostic
                    mapper_metrics["inference_diagnostic_path"] = diagnostic_path
            else:
                if runtime.load_checkpoint_path is None:
                    train_semantic(
                        model,
                        train_loader,
                        validation_loader,
                        runtime,
                        sid_to_items=sid_to_items,
                        sid_catalog=sid_catalog,
                        sid_logq_table=sid_logq_table,
                        tracker=tracker,
                    )
                else:
                    self._load_generator_checkpoint(
                        model,
                        runtime.load_checkpoint_path,
                        runtime.device,
                        runtime,
                    )
                validation_metrics = self._validate_global(
                    model,
                    validation_loader,
                    runtime,
                    sid_to_items,
                    sid_catalog,
                    sid_logq_table,
                )
                metrics = self._validate_global(
                    model,
                    test_loader,
                    runtime,
                    sid_to_items,
                    sid_catalog,
                    sid_logq_table,
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
            "checkpoint": (
                runtime.load_checkpoint_path
                if runtime.load_checkpoint_path is not None
                else runtime.checkpoint_path
            ),
            "loaded_checkpoint": runtime.load_checkpoint_path,
        }

    @staticmethod
    def _load_generator_checkpoint(
        model,
        checkpoint_path: Path,
        device: str,
        runtime: GenerationConfig,
    ) -> None:
        source = Path(checkpoint_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"generator checkpoint does not exist: {source}")
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state_dict"),
            dict,
        ):
            raise ValueError("generator checkpoint must contain model_state_dict")
        saved_config = checkpoint.get("config")
        if saved_config is not None:
            architecture_fields = (
                "max_history_len",
                "d_model",
                "d_ff",
                "num_heads",
                "num_layers",
                "use_user_profile",
                "user_profile_dim",
                "num_levels",
                "codebook_sizes",
            )
            for name in architecture_fields:
                saved_value = (
                    saved_config.get(name)
                    if isinstance(saved_config, dict)
                    else getattr(saved_config, name, None)
                )
                current_value = getattr(runtime, name)
                if name == "codebook_sizes" and saved_value is not None:
                    saved_value = tuple(saved_value)
                if saved_value != current_value:
                    raise ValueError(
                        "generator checkpoint has incompatible config field: " f"{name}"
                    )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    @staticmethod
    def _validate_reference_artifacts(
        artifact_dir: Path,
        *,
        semantic_df: pl.DataFrame,
        fit_user_sids: pl.DataFrame | None,
        observed_user_sids: pl.DataFrame | None,
    ) -> None:
        root = Path(artifact_dir).expanduser()
        expected = {
            root / "generation" / "semantic_ids.parquet": semantic_df,
        }
        if fit_user_sids is not None:
            expected[root / "user_semantic_ids_fit.parquet"] = fit_user_sids
        if observed_user_sids is not None:
            expected[root / "generation" / "user_semantic_ids_observed.parquet"] = (
                observed_user_sids
            )
        for path, current in expected.items():
            if not path.is_file():
                raise FileNotFoundError(f"reference artifact does not exist: {path}")
            reference = pl.read_parquet(path)
            columns = sorted(current.columns)
            if set(reference.columns) != set(columns) or not reference.select(
                columns
            ).sort(columns).equals(current.select(columns).sort(columns)):
                raise ValueError(f"reference artifact does not match: {path}")

    @staticmethod
    def _validate_global(
        model,
        loader,
        runtime: GenerationConfig,
        sid_to_items,
        sid_catalog,
        sid_logq_table=None,
    ) -> dict[str, Any]:
        if runtime.resolve_collisions:
            return validate_semantic_model_without_collisions(
                model,
                loader,
                runtime,
                sid_catalog=sid_catalog,
                sid_to_items=sid_to_items,
                sid_logq_table=sid_logq_table,
            )
        return validate_semantic_model_with_collisions(
            model,
            loader,
            runtime,
            sid_to_items,
            sid_catalog=sid_catalog,
            sid_logq_table=sid_logq_table,
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
        observed_user_sids: pl.DataFrame,
        sid_to_items: dict[tuple[int, ...], list[int]],
        validation_examples: list[dict],
        test_examples: list[dict],
    ) -> tuple[
        IUserSidMapper,
        dict[str, Any],
        dict[str, IUserSidMapper],
    ]:
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
                    content_sid_levels=runtime.rq_num_levels,
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
            diagnostics = None
            comparison_mappers: dict[str, IUserSidMapper] = {}
            if runtime.run_diagnostics:
                if runtime.mapper.kind == "frequency":
                    frequency_mapper = mapper
                else:
                    frequency_mapper = FrequencyUserSidMapper()
                    frequency_mapper.fit(mapper_training_frame)
                observed_mapping_frame = observed_user_sids.join(
                    semantic_df.select("item_id", *global_sid_columns),
                    on="item_id",
                    how="inner",
                )
                observed_oracle_mapper = FrequencyUserSidMapper()
                observed_oracle_mapper.fit(observed_mapping_frame)
                diagnostics = {
                    "ambiguity": {
                        "fit": mapping_ambiguity_metrics(
                            mapper_training_frame,
                            top_k=runtime.mapper_top_k,
                        ),
                        "observed": mapping_ambiguity_metrics(
                            observed_mapping_frame,
                            top_k=runtime.mapper_top_k,
                        ),
                    },
                    "fit_support": {
                        "validation": mapping_fit_support_metrics(
                            mapper_training_frame,
                            validation_examples,
                        ),
                        "test": mapping_fit_support_metrics(
                            mapper_training_frame,
                            test_examples,
                        ),
                    },
                    "frequency": {
                        "validation": frequency_mapper.evaluate_examples(
                            validation_examples,
                            k=runtime.mapper_top_k,
                        ),
                        "test": frequency_mapper.evaluate_examples(
                            test_examples,
                            k=runtime.mapper_top_k,
                        ),
                    },
                    "observed_oracle": {
                        "validation": observed_oracle_mapper.evaluate_examples(
                            validation_examples,
                            k=runtime.mapper_top_k,
                        ),
                        "test": observed_oracle_mapper.evaluate_examples(
                            test_examples,
                            k=runtime.mapper_top_k,
                        ),
                    },
                }
                comparison_mappers = {
                    "fit_frequency_mapper": frequency_mapper,
                    "observed_oracle_mapper": observed_oracle_mapper,
                }
            tracker.log_metrics(
                flatten_metrics(
                    {
                        "final.validation": validation,
                        "final.test": test,
                        "diagnostics": diagnostics,
                    }
                )
            )
        return (
            mapper,
            {
                "type": runtime.mapper.kind,
                "training": training,
                "validation": validation,
                "test": test,
                "diagnostics": diagnostics,
                "checkpoint": checkpoint,
            },
            comparison_mappers,
        )
