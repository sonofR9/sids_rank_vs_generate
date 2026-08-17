from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ..data.embedding_dataset import EmbeddingDataset
from .base import ISidPipeline
from .config import RQVAELetterConfig
from .rqvae_letter import RQVAE
from .rq_kmeans import RQKMeans


class RQVAELetterPipeline(ISidPipeline):
    def __init__(self, config: RQVAELetterConfig):
        self.config = config
        self.model = None
        self.requires_collaborative_embeddings = True

    def encode(self, X):
        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=str(self.model.device).startswith("cuda"),
        )

        self.model.eval()
        result = []

        with torch.no_grad():
            for x in loader:
                x = x.to(self.model.device, non_blocking=True)
                result.append(self.model.encoder(x).cpu())

        return torch.cat(result).numpy()

    def train(self, df: pl.DataFrame, *, collaborative_embeddings: pl.DataFrame):
        X = RQVAE.converter(df)

        collaborative_embeddings = (
            df.select("item_id")
            .join(
                collaborative_embeddings.select(
                    "item_id",
                    self.config.collaborative_embedding_column,
                ),
                on="item_id",
                how="left",
            )
            .rename({
                self.config.collaborative_embedding_column: "collaborative_embed"
            })
        )

        self.model = RQVAE(
            dim=X.shape[1],
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            codebook_size=self.config.codebook_size,
            codebook_sizes=(self.config.codebook_size,) * self.config.num_levels,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            lr=self.config.lr,
            beta=self.config.beta,
            collaborative_weight=self.config.collaborative_weight,
            diversity_weight=self.config.diversity_weight,
            diversity_clusters=self.config.diversity_clusters,
            num_workers=self.config.num_workers,
            device=self.config.device,
            pretrain_epochs=self.config.pretrain_epochs,
            pretrain_lr=self.config.pretrain_lr,
        )

        pretrain_loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=str(self.model.device).startswith("cuda"),
        )

        self.model.pretrain_autoencoder(
            pretrain_loader,
            tracker=self.tracker,
        )

        latent = self.encode(X)

        rqkmeans = RQKMeans(
            num_levels=self.config.num_levels,
            codebook_size=self.config.codebook_size,
            codebook_sizes=(self.config.codebook_size,) * self.config.num_levels,
            device=self.config.device,
        )

        rqkmeans.fit(latent, tracker=self.tracker)

        self.model.init_codebooks(rqkmeans.codebooks)

        self.model.fit(
            df,
            collaborative_embeddings,
            tracker=self.tracker,
        )

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.model.transform(df)

    def summary(self):
        return {
            "training_metrics": self.model.training_metrics,
        }
        