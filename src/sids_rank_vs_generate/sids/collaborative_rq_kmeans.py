from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as functional
from torch import nn

from ..reproducibility import seed_everything
from .base import ISidPipeline, SidTrainingData
from .config import (
    CollaborativeAlignmentConfig,
    CollaborativeAwareRQKMeansConfig,
)
from .rq_kmeans import RQKMeans


PAIR_COLUMNS = ("item_i", "item_j", "support")


def _require_interaction_columns(interactions: pl.DataFrame) -> None:
    required = {"uid", "item_id", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(
            "interactions are missing columns: " + ", ".join(sorted(missing))
        )
    if interactions.is_empty():
        raise ValueError("interactions must not be empty")
    if any(interactions[column].null_count() for column in required):
        raise ValueError("interactions contain nulls")


def mine_collaborative_pairs(
    interactions: pl.DataFrame,
    config: CollaborativeAlignmentConfig,
) -> pl.DataFrame:
    """Mine deterministic train-only item pairs from recent positive histories.

    Each unique item in a user's history is connected to the preceding
    ``history_window`` unique items. Pair support is the number of distinct
    users contributing that pair. The strongest pairs incident to each item
    are retained, which prevents popular items from consuming the whole pair
    budget.
    """

    _require_interaction_columns(interactions)
    recent_unique = (
        interactions.select("uid", "item_id", "timestamp")
        .sort(["uid", "timestamp", "item_id"])
        .unique(subset=["uid", "item_id"], keep="last", maintain_order=True)
        .sort(["uid", "timestamp", "item_id"])
    )
    support_by_lag: list[pl.DataFrame] = []
    for lag in range(1, config.history_window + 1):
        lagged = (
            recent_unique.with_columns(
                pl.col("item_id").shift(lag).over("uid").alias("_previous_item")
            )
            .filter(
                pl.col("_previous_item").is_not_null()
                & (pl.col("item_id") != pl.col("_previous_item"))
            )
            .with_columns(
                pl.min_horizontal("item_id", "_previous_item").alias("item_i"),
                pl.max_horizontal("item_id", "_previous_item").alias("item_j"),
            )
            .select("uid", "item_i", "item_j")
        )
        if not lagged.is_empty():
            # A user has one position per item after ``recent_unique``. The
            # distance of an unordered pair is therefore unique, so summing
            # per-lag user counts below still counts every user exactly once
            # without retaining O(rows * history_window) candidate rows.
            support_by_lag.append(
                lagged.group_by("item_i", "item_j").agg(
                    pl.len().cast(pl.Int64).alias("support")
                )
            )

    if not support_by_lag:
        raise ValueError(
            "no collaborative pairs could be mined from train interactions"
        )
    scored = (
        pl.concat(support_by_lag)
        .group_by("item_i", "item_j")
        .agg(pl.col("support").sum())
        .filter(pl.col("support") >= config.min_pair_support)
    )
    if scored.is_empty():
        raise ValueError(
            "no collaborative pairs satisfy min_pair_support="
            f"{config.min_pair_support}"
        )

    directed = pl.concat(
        [
            scored.select(
                pl.col("item_i").alias("anchor"),
                pl.col("item_j").alias("positive"),
                "support",
            ),
            scored.select(
                pl.col("item_j").alias("anchor"),
                pl.col("item_i").alias("positive"),
                "support",
            ),
        ]
    ).sort(
        ["anchor", "support", "positive"],
        descending=[False, True, False],
    )
    strongest = directed.group_by("anchor", maintain_order=True).head(
        config.pairs_per_item
    )
    selected = (
        strongest.with_columns(
            pl.min_horizontal("anchor", "positive").alias("item_i"),
            pl.max_horizontal("anchor", "positive").alias("item_j"),
        )
        .select("item_i", "item_j")
        .unique()
        .join(scored, on=["item_i", "item_j"], how="inner", validate="1:1")
        .sort(["support", "item_i", "item_j"], descending=[True, False, False])
    )
    if config.max_pairs is not None:
        selected = selected.head(config.max_pairs)
    if selected.height < 2:
        raise ValueError("contrastive alignment requires at least two item pairs")
    return selected.select(*PAIR_COLUMNS)


class _ResidualAlignment(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )
        output = self.projection[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.normalize(values + self.projection(values), dim=-1)


def _embedding_matrix(
    item_features: pl.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in item_features.columns:
        raise ValueError(f"item features must contain {column!r}")
    matrix = np.asarray(item_features[column].to_list(), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("item embeddings must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("item embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("item embeddings must not contain zero vectors")
    return matrix / norms


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CollaborativeAwareRQKMeansPipeline(ISidPipeline):
    """OneRec-style collaborative alignment followed by residual K-Means."""

    def __init__(
        self,
        config: CollaborativeAwareRQKMeansConfig | None = None,
    ) -> None:
        self.config = config or CollaborativeAwareRQKMeansConfig()
        self._alignment_model: _ResidualAlignment | None = None
        self._rq_kmeans: RQKMeans | None = None
        self._device: torch.device | None = None
        self._summary: dict[str, Any] | None = None

    def train(self, df: pl.DataFrame) -> None:
        del df
        raise RuntimeError(
            "CollaborativeAwareRQKMeansPipeline requires train interactions; "
            "use train_with_context() or the shared run.py experiment runner"
        )

    @staticmethod
    def _pair_indices(
        item_features: pl.DataFrame,
        pairs: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        item_index = item_features.select("item_id").with_row_index("_index")
        indexed = (
            pairs.join(
                item_index.rename({"item_id": "item_i", "_index": "_index_i"}),
                on="item_i",
                how="left",
                validate="m:1",
            )
            .join(
                item_index.rename({"item_id": "item_j", "_index": "_index_j"}),
                on="item_j",
                how="left",
                validate="m:1",
            )
        )
        if indexed["_index_i"].null_count() or indexed["_index_j"].null_count():
            raise ValueError("collaborative pairs contain items outside the catalog")
        return (
            indexed["_index_i"].cast(pl.Int64).to_numpy(),
            indexed["_index_j"].cast(pl.Int64).to_numpy(),
        )

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

    def _fit_alignment(self, data: SidTrainingData) -> np.ndarray:
        alignment = self.config.alignment
        seed_everything(alignment.seed)
        pairs = mine_collaborative_pairs(data.train_interactions, alignment)
        matrix = _embedding_matrix(data.item_features, alignment.embedding_column)
        index_i, index_j = self._pair_indices(data.item_features, pairs)
        device = _resolve_device(alignment.device)
        model = _ResidualAlignment(matrix.shape[1], alignment.hidden_dim).to(device)
        values = torch.from_numpy(matrix).to(device)
        pair_i = torch.from_numpy(index_i.astype(np.int64, copy=True))
        pair_j = torch.from_numpy(index_j.astype(np.int64, copy=True))
        support = torch.from_numpy(
            pairs["support"].to_numpy().astype(np.float32, copy=True)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=alignment.learning_rate,
            weight_decay=alignment.weight_decay,
        )
        generator = torch.Generator(device="cpu").manual_seed(alignment.seed)
        epoch_losses: list[float] = []
        epoch_contrastive_losses: list[float] = []
        epoch_preservation_losses: list[float] = []
        model.train()
        for epoch in range(alignment.epochs):
            permutation = torch.randperm(pairs.height, generator=generator)
            totals = np.zeros(3, dtype=np.float64)
            examples = 0
            for offset in range(0, pairs.height, alignment.batch_size):
                selected = permutation[offset : offset + alignment.batch_size]
                if selected.numel() < 2:
                    continue
                selected_device = selected.to(device)
                original_i = values[pair_i[selected].to(device)]
                original_j = values[pair_j[selected].to(device)]
                encoded_i = model(original_i)
                encoded_j = model(original_j)
                logits = encoded_i @ encoded_j.transpose(0, 1)
                logits = logits / alignment.temperature
                targets = torch.arange(logits.shape[0], device=device)
                per_pair = 0.5 * (
                    functional.cross_entropy(logits, targets, reduction="none")
                    + functional.cross_entropy(
                        logits.transpose(0, 1), targets, reduction="none"
                    )
                )
                weights = torch.log1p(support[selected]).to(device)
                weights = weights / weights.mean()
                contrastive_loss = (per_pair * weights).mean()
                preservation_loss = 0.5 * (
                    (1.0 - (encoded_i * original_i).sum(dim=1)).mean()
                    + (1.0 - (encoded_j * original_j).sum(dim=1)).mean()
                )
                loss = contrastive_loss + (
                    alignment.content_preservation_weight * preservation_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = selected.numel()
                totals += np.asarray(
                    [loss.item(), contrastive_loss.item(), preservation_loss.item()]
                ) * batch_size
                examples += batch_size
            if examples == 0:
                raise ValueError(
                    "contrastive alignment produced no batch with at least two pairs"
                )
            averages = totals / examples
            epoch_losses.append(float(averages[0]))
            epoch_contrastive_losses.append(float(averages[1]))
            epoch_preservation_losses.append(float(averages[2]))
            self.tracker.log_metrics(
                {
                    "alignment.loss": averages[0],
                    "alignment.contrastive_loss": averages[1],
                    "alignment.content_preservation_loss": averages[2],
                },
                step=epoch,
            )

        self._alignment_model = model.eval()
        self._device = device
        aligned = self._align_matrix(matrix)

        before_similarity = np.sum(matrix[index_i] * matrix[index_j], axis=1)
        after_similarity = np.sum(aligned[index_i] * aligned[index_j], axis=1)
        content_similarity = np.sum(matrix * aligned, axis=1)
        self._summary = {
            "pair_mining": {
                "source": "train_interactions",
                "train_end_timestamp": str(data.train_end_timestamp),
                "interaction_count": data.train_interactions.height,
                "pair_count": pairs.height,
                "mean_support": float(pairs["support"].mean()),
                "max_support": int(pairs["support"].max()),
            },
            "alignment": {
                "initial_loss": epoch_losses[0],
                "final_loss": epoch_losses[-1],
                "final_contrastive_loss": epoch_contrastive_losses[-1],
                "final_content_preservation_loss": epoch_preservation_losses[-1],
                "mean_pair_cosine_before": float(before_similarity.mean()),
                "mean_pair_cosine_after": float(after_similarity.mean()),
                "mean_content_cosine": float(content_similarity.mean()),
                "device": str(device),
            },
        }
        self.tracker.log_metrics(
            {
                "pair_mining.pair_count": pairs.height,
                "pair_mining.mean_support": float(pairs["support"].mean()),
                "alignment.mean_pair_cosine_before": before_similarity.mean(),
                "alignment.mean_pair_cosine_after": after_similarity.mean(),
                "alignment.mean_content_cosine": content_similarity.mean(),
            }
        )
        return aligned

    def train_with_context(self, data: SidTrainingData) -> None:
        aligned = self._fit_alignment(data)
        rq_kmeans = self._build_rq_kmeans()
        rq_kmeans.fit(aligned)
        self._rq_kmeans = rq_kmeans

    @torch.no_grad()
    def _align_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self._alignment_model is None or self._device is None:
            raise RuntimeError("train_with_context() must be called before alignment")
        batch_size = self.config.alignment.inference_batch_size
        outputs: list[np.ndarray] = []
        for offset in range(0, len(matrix), batch_size):
            batch = torch.from_numpy(matrix[offset : offset + batch_size]).to(
                self._device
            )
            outputs.append(self._alignment_model(batch).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32, copy=False)

    def aligned_embeddings(self, item_features: pl.DataFrame) -> np.ndarray:
        matrix = _embedding_matrix(
            item_features,
            self.config.alignment.embedding_column,
        )
        return self._align_matrix(matrix)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._rq_kmeans is None:
            raise RuntimeError("train_with_context() must be called before update()")
        aligned = self.aligned_embeddings(df)
        codes = self._rq_kmeans.get_semantic_ids(aligned)
        return pl.DataFrame(
            {
                "item_id": df["item_id"],
                **{
                    f"sid_{level}": codes[:, level]
                    for level in range(codes.shape[1])
                },
            }
        )

    def summary(self) -> dict[str, Any] | None:
        return self._summary
