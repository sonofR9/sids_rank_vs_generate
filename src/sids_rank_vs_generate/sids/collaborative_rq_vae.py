from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import polars as pl

from ..tracking import ExperimentTracker
from .base import SidTrainingData
from .collaborative_rq_kmeans import CollaborativeAwareRQKMeansPipeline
from .config import CollaborativeAwareRQVAEConfig, RQVAEConfig
from .rq_vae_pipeline import RQVAEPipeline


class CollaborativeAwareRQVAEPipeline(CollaborativeAwareRQKMeansPipeline):
    """OneRec-style pair alignment followed by an RQ-VAE tokenizer."""

    def __init__(
        self,
        config: CollaborativeAwareRQVAEConfig | None = None,
    ) -> None:
        self.config = config or CollaborativeAwareRQVAEConfig()
        self._alignment_model = None
        self._device = None
        self._summary = None
        rqvae_config = RQVAEConfig(
            **{
                field.name: getattr(self.config, field.name)
                for field in fields(RQVAEConfig)
            }
        )
        self._rqvae = RQVAEPipeline(rqvae_config)

    def set_tracker(self, tracker: ExperimentTracker) -> None:
        super().set_tracker(tracker)
        self._rqvae.set_tracker(tracker)

    def train(self, df: pl.DataFrame) -> None:
        del df
        raise RuntimeError(
            "CollaborativeAwareRQVAEPipeline requires train interactions; "
            "use train_with_context() or the shared run.py experiment runner"
        )

    def _replace_embeddings(
        self,
        frame: pl.DataFrame,
        aligned: np.ndarray,
    ) -> pl.DataFrame:
        if len(aligned) != frame.height:
            raise ValueError("aligned embedding count must match item features")
        return frame.with_columns(
            pl.Series("normalized_embed", aligned.tolist())
        )

    def train_with_context(self, data: SidTrainingData) -> None:
        aligned = self._fit_alignment(data)
        self._rqvae.train(self._replace_embeddings(data.item_features, aligned))

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        aligned = self.aligned_embeddings(df)
        return self._rqvae.update(self._replace_embeddings(df, aligned))

    def summary(self) -> dict[str, Any]:
        alignment = self._summary or {}
        return {
            **alignment,
            "tokenizer": {
                "type": "rqvae",
                "assignment": self.config.assignment_method,
                "details": self._rqvae.summary(),
            },
        }
