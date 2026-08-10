from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ..data.embedding_dataset import EmbeddingDataset
from .base import ISidPipeline
from .config import AssignmentMethod, RQVAEConfig
from .pure_semantic import allocate_semantic_ids
from .rq_kmeans import RQKMeans
from .rq_vae import RQVAE
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything


class RQVAEPipeline(ISidPipeline):
    def __init__(self, config: RQVAEConfig | None = None) -> None:
        self.config = config or RQVAEConfig()
        self.model: RQVAE | None = None
        self.assignment_metrics: dict[str, float | int | str] | None = None

    @torch.no_grad()
    def _encode_latents(self, frame: pl.DataFrame) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("train() must be called before encoding")
        values = self.model.converter(frame)
        loader = DataLoader(
            EmbeddingDataset(values),
            batch_size=self.model.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=str(self.model.device).startswith("cuda"),
            num_workers=self.model.num_workers,
        )
        self.model.eval()
        latents = [
            self.model.encoder(batch.to(self.model.device, non_blocking=True))
            for batch in loader
        ]
        return torch.cat(latents, dim=0)

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(SID_EXPERIMENT_SEED)
        parameters = asdict(self.config)
        parameters.pop("assignment_method")
        parameters.pop("psi_top_k")
        dimension = len(df["normalized_embed"][0])
        self.model = RQVAE(dim=dimension, **parameters)
        values = self.model.converter(df)
        loader = DataLoader(
            EmbeddingDataset(values),
            batch_size=self.model.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=str(self.model.device).startswith("cuda"),
            num_workers=self.model.num_workers,
        )
        self.model.pretrain_autoencoder(loader, tracker=self.tracker)
        latent = self._encode_latents(df).cpu().numpy()
        initializer = RQKMeans(
            num_levels=self.config.num_levels,
            codebook_size=self.config.codebook_size,
        ).fit(latent)
        self.model.init_codebooks(initializer.codebooks)
        self.model.fit(values, tracker=self.tracker)
        self.assignment_metrics = None

    def assign(
        self,
        df: pl.DataFrame,
        *,
        method: AssignmentMethod | None = None,
    ) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("train() must be called before assign()")
        allocation = allocate_semantic_ids(
            self._encode_latents(df),
            tuple(self.model.codebooks),
            method=method or self.config.assignment_method,
            top_k=self.config.psi_top_k,
        )
        self.assignment_metrics = allocation.metrics
        self.tracker.log_metrics(
            {
                f"assignment.{name}": value
                for name, value in allocation.metrics.items()
                if isinstance(value, (int, float))
            }
        )
        codes = allocation.codes.cpu().numpy().astype(np.int32)
        return pl.DataFrame(
            {
                "item_id": df["item_id"],
                **{
                    f"sid_{level}": codes[:, level]
                    for level in range(self.model.num_levels)
                },
            }
        )

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.assign(df)

    def summary(self) -> dict[str, Any]:
        return {"assignment": self.assignment_metrics}
