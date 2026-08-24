from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import torch

from ..models.ranker_pair_encoder import RankerPairEncoder
from ..training.ranker_pair_encoder import (
    RankerPairEncoderTrainingConfig,
    encode_ranker_pairs_indexed,
    fit_ranker_pair_encoder,
)
from .rq_kmeans import RQKMeans
from .user_aware import UserAwareSidTrainingData
from .user_aware_rq_kmeans import UserAwareRQKMeansPipeline


def _default_ranker_training() -> RankerPairEncoderTrainingConfig:
    return RankerPairEncoderTrainingConfig(
        batch_size=8_192,
        epochs=8,
        precision="fp32",
        device="auto",
    )


@dataclass(frozen=True)
class RankerEmbeddingUserAwareRQKMeansConfig:
    """Настройки отдельного user-aware 2.0 pipeline"""

    num_levels: int | None = 3
    codebook_size: int | None = 256
    codebook_sizes: tuple[int, ...] | None = None
    embedding_column: str = "normalized_embed"
    user_profile_weight: float = 1.0
    kmeans_batch_size: int = 16_384
    predict_batch_size: int = 65_536
    ranker_hidden_dims: tuple[int, ...] = (256, 128)
    ranker_representation_dim: int = 64
    ranker_dropout: float = 0.1
    fusion_weight: float = 0.35
    ranker_label_fraction: float = 0.2
    negative_ratio: int = 10
    ranker_training: RankerPairEncoderTrainingConfig = field(
        default_factory=_default_ranker_training
    )

    def __post_init__(self) -> None:
        if self.codebook_sizes is not None:
            if not self.codebook_sizes or any(
                size <= 0 for size in self.codebook_sizes
            ):
                raise ValueError("codebook_sizes must contain positive values")
        elif self.num_levels is None or self.codebook_size is None:
            raise ValueError("num_levels and codebook_size are required")
        elif self.num_levels <= 0 or self.codebook_size <= 0:
            raise ValueError("num_levels and codebook_size must be positive")
        if not self.embedding_column:
            raise ValueError("embedding_column must not be empty")
        if self.user_profile_weight < 0:
            raise ValueError("user_profile_weight must be non-negative")
        if self.kmeans_batch_size <= 0 or self.predict_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not self.ranker_hidden_dims or any(
            width <= 0 for width in self.ranker_hidden_dims
        ):
            raise ValueError("ranker_hidden_dims must contain positive values")
        if self.ranker_representation_dim <= 0:
            raise ValueError("ranker_representation_dim must be positive")
        if not 0.0 <= self.ranker_dropout < 1.0:
            raise ValueError("ranker_dropout must be in [0, 1)")
        if not 0.0 <= self.fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        if not 0.0 < self.ranker_label_fraction < 1.0:
            raise ValueError("ranker_label_fraction must be in (0, 1)")
        if (
            not isinstance(self.negative_ratio, int)
            or isinstance(self.negative_ratio, bool)
            or self.negative_ratio <= 0
        ):
            raise ValueError("negative_ratio must be a positive integer")


class RankerEmbeddingUserAwareRQKMeansPipeline(UserAwareRQKMeansPipeline):
    """User-aware SID с отдельным fit-only pair encoder"""

    def __init__(
        self,
        config: RankerEmbeddingUserAwareRQKMeansConfig | None = None,
    ) -> None:
        super().__init__(config or RankerEmbeddingUserAwareRQKMeansConfig())
        self.config: RankerEmbeddingUserAwareRQKMeansConfig
        self._ranker_encoder: RankerPairEncoder | None = None
        self._item_embedding_table: torch.Tensor | None = None
        self._user_profile_table: torch.Tensor | None = None
        self._item_index: pl.DataFrame | None = None
        self._user_profile_index: pl.DataFrame | None = None
        self._unknown_profile_index: int | None = None
        self.training_summary: dict[str, Any] = {}

    @property
    def requires_behavior_interactions(self) -> bool:
        return True

    @staticmethod
    def _require_behavior_frame(data: UserAwareSidTrainingData) -> pl.DataFrame:
        behavior = getattr(data, "fit_behavior_interactions", None)
        if behavior is None:
            raise ValueError(
                "fit_behavior_interactions is required for user-aware 2.0"
            )
        if not isinstance(behavior, pl.DataFrame):
            raise TypeError("fit_behavior_interactions must be a Polars DataFrame")
        required = {"uid", "item_id", "timestamp", "target"}
        missing = required - set(behavior.columns)
        if missing:
            raise ValueError(
                "fit_behavior_interactions is missing columns: "
                + ", ".join(sorted(missing))
            )
        if behavior.is_empty():
            raise ValueError("fit_behavior_interactions must not be empty")
        nulls = behavior.select(
            pl.col(column).null_count().alias(column) for column in required
        ).row(0, named=True)
        if any(nulls.values()):
            raise ValueError("fit_behavior_interactions must not contain nulls")
        targets = set(behavior["target"].cast(pl.Float32).unique().to_list())
        if not targets.issubset({0.0, 1.0}):
            raise ValueError("fit behavior target must be binary")
        return behavior.select(
            "uid",
            "item_id",
            "timestamp",
            pl.col("target").cast(pl.Float32),
        )

    def _ranker_cutoff(self, fit_interactions: pl.DataFrame) -> Any:
        timestamps = fit_interactions["timestamp"].drop_nulls()
        if timestamps.is_empty():
            raise ValueError("fit interactions must contain non-null timestamps")
        cutoff = timestamps.quantile(
            1.0 - self.config.ranker_label_fraction,
            interpolation="nearest",
        )
        if cutoff is None:
            raise ValueError("could not determine ranker cutoff")
        return cutoff

    def _sample_ranker_labels(
        self,
        behavior: pl.DataFrame,
        *,
        cutoff: Any,
        fit_end_timestamp: Any,
    ) -> pl.DataFrame:
        if behavior["timestamp"].max() >= fit_end_timestamp:
            raise ValueError(
                "fit_behavior_interactions must be strictly earlier than "
                "fit_end_timestamp"
            )
        window = behavior.filter(
            (pl.col("timestamp") >= pl.lit(cutoff))
            & (pl.col("timestamp") < pl.lit(fit_end_timestamp))
        )
        positives = window.filter(pl.col("target") == 1.0)
        negatives = window.filter(pl.col("target") == 0.0)
        if positives.is_empty() or negatives.is_empty():
            raise ValueError("ranker label window must contain both classes")

        negative_limit = min(
            negatives.height,
            positives.height * self.config.negative_ratio,
        )
        hash_columns = ["uid", "item_id", "timestamp", "target"]
        negatives = (
            negatives.with_columns(
                pl.struct(hash_columns)
                .hash(seed=self.config.ranker_training.seed)
                .alias("_stable_hash")
            )
            .sort(["_stable_hash", "timestamp", "uid", "item_id"])
            .head(negative_limit)
            .drop("_stable_hash")
        )
        return pl.concat([positives, negatives]).sort(
            ["timestamp", "uid", "item_id", "target"]
        )

    def _refresh_index_tables(self) -> None:
        if self._unknown_user_profile is None:
            raise RuntimeError("user profiles are not fitted")
        item_ids = list(self._item_embeddings)
        user_ids = list(self._user_profiles)
        self._item_embedding_table = torch.from_numpy(
            np.stack([self._item_embeddings[item_id] for item_id in item_ids])
            .astype(np.float32)
        )
        self._user_profile_table = torch.from_numpy(
            np.stack(
                [self._user_profiles[uid] for uid in user_ids]
                + [self._unknown_user_profile]
            ).astype(np.float32)
        )
        self._item_index = pl.DataFrame({"item_id": item_ids}).with_row_index(
            "_item_index"
        )
        self._user_profile_index = pl.DataFrame({"uid": user_ids}).with_row_index(
            "_profile_index"
        )
        self._unknown_profile_index = len(user_ids)

    def _indexed_rows(
        self,
        rows: pl.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._item_index is None
            or self._user_profile_index is None
            or self._unknown_profile_index is None
        ):
            raise RuntimeError("index tables are not initialized")
        indexed = (
            rows.select("uid", "item_id")
            .with_row_index("_row_index")
            .join(self._item_index, on="item_id", how="left", validate="m:1")
        )
        if indexed["_item_index"].null_count():
            raise ValueError("pairs contain items outside the fitted catalog")
        indexed = (
            indexed.join(
                self._user_profile_index,
                on="uid",
                how="left",
                validate="m:1",
            )
            .with_columns(
                pl.col("_profile_index").fill_null(self._unknown_profile_index)
            )
            .sort("_row_index")
        )
        return (
            torch.from_numpy(
                indexed["_item_index"].cast(pl.Int64).to_numpy().copy()
            ),
            torch.from_numpy(
                indexed["_profile_index"].cast(pl.Int64).to_numpy().copy()
            ),
        )

    def _encode_ranker_features(
        self,
        pairs: pl.DataFrame,
        *,
        indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> np.ndarray:
        if (
            self._ranker_encoder is None
            or self._item_embedding_table is None
            or self._user_profile_table is None
        ):
            raise RuntimeError("train() must be called before encoding pairs")
        item_indices, profile_indices = indices or self._indexed_rows(pairs)
        encoded = encode_ranker_pairs_indexed(
            self._ranker_encoder,
            self._item_embedding_table,
            self._user_profile_table,
            item_indices,
            profile_indices,
            batch_size=self.config.predict_batch_size,
            device=self.config.ranker_training.device,
            precision=self.config.ranker_training.precision,
            table_placement=self.config.ranker_training.table_placement,
            max_device_table_fraction=(
                self.config.ranker_training.max_device_table_fraction
            ),
        )
        return encoded.numpy()

    def _pair_features(self, pairs: pl.DataFrame) -> np.ndarray:
        if pairs.is_empty():
            raise ValueError("pairs must not be empty")
        if self._item_embedding_table is None or self._user_profile_table is None:
            raise RuntimeError("train() must be called before encoding pairs")
        item_indices, profile_indices = self._indexed_rows(pairs)
        items = self._item_embedding_table[item_indices].numpy()
        users = (
            self._user_profile_table[profile_indices].numpy()
            * self.config.user_profile_weight
        )
        base = self._normalize_rows(
            np.concatenate(
                [items, users, items * users, np.abs(items - users)],
                axis=1,
            ).astype(np.float32)
        )
        collaborative = self._encode_ranker_features(
            pairs,
            indices=(item_indices, profile_indices),
        )
        weight = self.config.fusion_weight
        fused = np.concatenate(
            [math.sqrt(1.0 - weight * weight) * base, weight * collaborative],
            axis=1,
        ).astype(np.float32)
        return self._normalize_rows(fused)

    def _build_rq_kmeans(self) -> RQKMeans:
        if self.config.codebook_sizes is not None:
            return RQKMeans(
                codebook_sizes=self.config.codebook_sizes,
                kmeans_batch_size=self.config.kmeans_batch_size,
                predict_batch_size=self.config.predict_batch_size,
            )
        return RQKMeans(
            num_levels=self.config.num_levels,
            codebook_size=self.config.codebook_size,
            kmeans_batch_size=self.config.kmeans_batch_size,
            predict_batch_size=self.config.predict_batch_size,
        )

    def train(self, data: UserAwareSidTrainingData) -> None:
        behavior = self._require_behavior_frame(data)
        cutoff = self._ranker_cutoff(data.fit_interactions)
        early_likes = data.fit_interactions.filter(
            pl.col("timestamp") < pl.lit(cutoff)
        )
        if early_likes.is_empty():
            raise ValueError("no profile interactions remain before ranker cutoff")

        self._load_item_embeddings(data.item_features.sort("item_id"))
        catalog = set(self._item_embeddings)
        unknown_behavior_items = (
            set(behavior["item_id"].unique().to_list()) - catalog
        )
        if unknown_behavior_items:
            raise ValueError("fit behavior contains items outside the catalog")

        sampled_labels = self._sample_ranker_labels(
            behavior,
            cutoff=cutoff,
            fit_end_timestamp=data.fit_end_timestamp,
        )
        self._fit_user_profiles(early_likes)
        self._refresh_index_tables()
        assert self._item_embedding_table is not None
        assert self._user_profile_table is not None
        assert self._unknown_profile_index is not None
        item_indices, profile_indices = self._indexed_rows(sampled_labels)
        unknown_profile_rows = int(
            (profile_indices == self._unknown_profile_index).sum().item()
        )
        unknown_profile_rate = unknown_profile_rows / sampled_labels.height

        self._ranker_encoder = RankerPairEncoder(
            embedding_dim=self._item_embedding_table.shape[1],
            hidden_dims=self.config.ranker_hidden_dims,
            representation_dim=self.config.ranker_representation_dim,
            dropout=self.config.ranker_dropout,
        )
        fit_result = fit_ranker_pair_encoder(
            self._ranker_encoder,
            self._item_embedding_table,
            self._user_profile_table,
            item_indices,
            profile_indices,
            torch.from_numpy(sampled_labels["target"].to_numpy().copy()),
            self.config.ranker_training,
        )

        for epoch, loss in enumerate(fit_result.epoch_losses, start=1):
            self.tracker.log_metrics(
                {"pair_encoder.train.loss": loss},
                step=epoch,
            )
        self.tracker.log_metrics(
            {
                f"pair_encoder.fit.{name}": value
                for name, value in fit_result.metrics.items()
            }
        )

        early_max_timestamp = early_likes["timestamp"].max()
        label_min_timestamp = sampled_labels["timestamp"].min()
        label_max_timestamp = sampled_labels["timestamp"].max()
        positive_count = int(sampled_labels["target"].sum())
        summary_metrics = {
            "pair_encoder.rows": sampled_labels.height,
            "pair_encoder.positives": positive_count,
            "pair_encoder.negatives": sampled_labels.height - positive_count,
            "pair_encoder.early_profile_rows": early_likes.height,
            "pair_encoder.early_profile_users": len(self._user_profiles),
            "pair_encoder.unknown_profile_rows": unknown_profile_rows,
            "pair_encoder.unknown_profile_rate": unknown_profile_rate,
        }
        self.tracker.log_metrics(summary_metrics)
        self.tracker.log_params(
            {
                "user_aware_version": "2.0",
                "pair_encoder.cutoff": cutoff,
                "pair_encoder.label_fraction": self.config.ranker_label_fraction,
                "pair_encoder.negative_ratio": self.config.negative_ratio,
                "pair_encoder.fusion_weight": self.config.fusion_weight,
                "pair_encoder.representation_dim": (
                    self.config.ranker_representation_dim
                ),
            }
        )

        self._fit_user_profiles(data.fit_interactions)
        self._refresh_index_tables()
        fit_pairs = (
            data.fit_interactions.select("uid", "item_id")
            .unique()
            .sort(["uid", "item_id"])
        )
        fused_features = self._pair_features(fit_pairs)
        self._model = self._build_rq_kmeans()
        self._model.fit(fused_features)

        self.training_summary = {
            "version": "2.0",
            "ranker_cutoff": cutoff,
            "ranker_profile_max_timestamp": early_max_timestamp,
            "ranker_label_min_timestamp": label_min_timestamp,
            "ranker_label_max_timestamp": label_max_timestamp,
            "ranker_rows": sampled_labels.height,
            "ranker_positives": positive_count,
            "ranker_negatives": sampled_labels.height - positive_count,
            "ranker_unknown_profile_rows": unknown_profile_rows,
            "ranker_unknown_profile_rate": unknown_profile_rate,
            "ranker_fit": fit_result.as_dict(),
            "full_profile_users": len(self._user_profiles),
            "fit_pairs": fit_pairs.height,
            "base_representation_dim": fused_features.shape[1]
            - self.config.ranker_representation_dim,
            "collaborative_representation_dim": (
                self.config.ranker_representation_dim
            ),
            "fused_representation_dim": fused_features.shape[1],
        }

    def summary(self) -> dict[str, Any]:
        return deepcopy(self.training_summary)
