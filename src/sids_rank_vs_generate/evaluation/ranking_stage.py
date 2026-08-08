from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..data.ranking import build_ranking_loaders, prepare_ranking_data
from ..models.ranking import DcnV2Ranker, TransActDcnV2Ranker
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from ..tracking import flatten_parameters
from ..training.ranking import train_ranking_model
from .stages import EvaluationContext, IEvaluationStage


@dataclass(frozen=True)
class RankingConfig:
    """Training hyperparameters for the two fixed ranking models."""

    max_history_len: int = 50
    min_history_len: int = 2
    validation_fraction: float = 0.1
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    embedding_dim: int = 32
    hidden_dim: int = 128
    dense_bins: int = 32
    dense_train_rows: int = 1_000_000
    dcn_cross_layers: int = 3
    dcn_deep_units: tuple[int, ...] = (256, 128, 64)
    dcn_num_experts: int = 4
    dcn_low_rank: int = 32
    transact_d_model: int = 64
    transact_heads: int = 4
    transact_layers: int = 1
    transact_ff_dim: int = 256
    dropout: float = 0.2
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    models: tuple[str, ...] = ("dcnv2", "transact")
    use_semantic_ids: bool = True
    use_bf16: bool = True
    show_progress: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.transact_d_model % self.transact_heads:
            raise ValueError("transact_d_model must be divisible by transact_heads")
        if self.min_history_len <= 0 or self.max_history_len < self.min_history_len:
            raise ValueError("invalid ranking history lengths")
        supported_models = {"dcnv2", "transact"}
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("ranking models must be non-empty and unique")
        if unknown_models := set(self.models) - supported_models:
            raise ValueError(f"unsupported ranking models: {sorted(unknown_models)}")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")


class RankingStage(IEvaluationStage):
    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    @property
    def name(self) -> str:
        return "ranking"

    def evaluate(self, context: EvaluationContext) -> dict[str, Any]:
        data = prepare_ranking_data(
            context.prepared,
            context.items_with_sids,
            context.dataset,
            max_history_len=self.config.max_history_len,
            min_history_len=self.config.min_history_len,
            validation_fraction=self.config.validation_fraction,
            max_rows=context.max_rows,
            user_sid_pipeline=context.user_sid_pipeline,
            use_semantic_ids=self.config.use_semantic_ids,
        )
        loaders = build_ranking_loaders(
            data,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )

        model_factories = {
            "dcnv2": DcnV2Ranker,
            "transact": TransActDcnV2Ranker,
        }
        results = {}
        for model_name in self.config.models:
            model_factory = model_factories[model_name]
            seed_everything(SID_EXPERIMENT_SEED)
            with context.tracker.start_run(
                f"ranking.{model_name}",
                nested=True,
                tags={"stage": "ranking", "model": model_name},
            ) as tracker:
                tracker.log_params(
                    flatten_parameters(
                        {
                            "config": asdict(self.config),
                            "examples": {
                                "train": data.train.height,
                                "validation": data.validation.height,
                                "test": data.test.height,
                            },
                        }
                    )
                )
                results[model_name] = train_ranking_model(
                    model_factory(data, self.config),
                    *loaders,
                    self.config,
                    context.artifact_dir / f"{model_name}.pt",
                    tracker=tracker,
                )

        return {
            "config": asdict(self.config),
            "features": {
                "dense": list(data.dense_train.columns),
                "sparse": ["uid", *data.sid_columns],
                "multivalent": ["artist_ids", "album_ids"],
                "transact_history": (
                    "preceding semantic-ID likes"
                    if self.config.use_semantic_ids
                    else "preceding item likes"
                ),
            },
            "examples": {
                "train": data.train.height,
                "validation": data.validation.height,
                "test": data.test.height,
            },
            "models": results,
        }
