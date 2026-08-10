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
    validation_days: float = 7.0
    hyperparameter_tuning: bool = False
    train_days: float = 60.0
    batch_size: int = 32_768
    epochs: int = 2
    learning_rate: float = 2e-3
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
    show_progress: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.validation_days <= 0:
            raise ValueError("validation_days must be positive")
        if self.train_days <= 0:
            raise ValueError("train_days must be positive")
        if self.transact_d_model % self.transact_heads:
            raise ValueError("transact_d_model must be divisible by transact_heads")
        if self.min_history_len <= 0 or self.max_history_len < self.min_history_len:
            raise ValueError("invalid ranking history lengths")


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
            validation_days=self.config.validation_days,
            hyperparameter_tuning=self.config.hyperparameter_tuning,
            train_days=self.config.train_days,
            max_rows=context.max_rows,
        )
        loaders = build_ranking_loaders(
            data,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        model_factories = {
            "dcnv2": DcnV2Ranker,
            "transact": TransActDcnV2Ranker,
        }
        results = {}
        for model_name, model_factory in model_factories.items():
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
                                "validation": (
                                    data.validation.height
                                    if data.validation is not None
                                    else 0
                                ),
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
            "primary_metric": "pairwise_accuracy",
            "features": {
                "dense": list(data.dense_train.columns),
                "sparse": ["uid", *data.sid_columns],
                "multivalent": ["artist_ids", "album_ids"],
                "transact_history": "preceding generation likes",
            },
            "examples": {
                "train": data.train.height,
                "validation": (
                    data.validation.height if data.validation is not None else 0
                ),
                "test": data.test.height,
            },
            "train_start_timestamp": str(data.train_start_timestamp),
            "train_end_timestamp": str(data.train_end_timestamp),
            "test_start_timestamp": str(data.test_start_timestamp),
            "models": results,
        }
