from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from .base import ISidPipeline
from .config import AssignmentMethod, RQVAEConfig
from .pure_semantic import allocate_semantic_ids
from .rq_kmeans import RQKMeans
from .rq_vae import RQVAE


class RQVAEPipeline(ISidPipeline):
    def __init__(self, config: RQVAEConfig | None = None) -> None:
        self.config = config or RQVAEConfig()
        self.model: RQVAE | None = None
        self.assignment_metrics: dict[str, float | int | str] | None = None
        self.checkpoint_loaded = False

    def _training_parameters(self) -> dict[str, Any]:
        parameters = asdict(self.config)
        for name in (
            "assignment_method",
            "psi_top_k",
            "checkpoint_path",
            "reuse_checkpoint",
        ):
            parameters.pop(name)
        return parameters

    def _checkpoint_signature(self) -> dict[str, Any]:
        parameters = self._training_parameters()
        parameters.pop("device")
        parameters.pop("num_workers")
        return parameters

    def _load_checkpoint(self, path: Path, *, dimension: int, item_count: int) -> bool:
        if not self.config.reuse_checkpoint or not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location=self.model.device, weights_only=True)
        except TypeError:  # PyTorch 2.2 compatibility.
            payload = torch.load(path, map_location=self.model.device)
        expected = {
            "format_version": 1,
            "embedding_dim": dimension,
            "item_count": item_count,
            "training_config": self._checkpoint_signature(),
        }
        actual = {name: payload.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"RQ-VAE checkpoint {path} does not match the current dataset/config"
            )
        self.model.load_state_dict(payload["model_state_dict"])
        return True

    def _save_checkpoint(self, path: Path, *, dimension: int, item_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "embedding_dim": dimension,
                "item_count": item_count,
                "training_config": self._checkpoint_signature(),
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    @torch.inference_mode()
    def _encode_latents(
        self,
        source: pl.DataFrame | torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("train() must be called before encoding")
        values = (
            source.to(self.model.device)
            if isinstance(source, torch.Tensor)
            else torch.from_numpy(self.model.converter(source)).to(
                self.model.device
            )
        )
        self.model.eval()
        latents = [
            self.model.encoder(batch)
            for batch in self.model._iter_batches(values, shuffle=False)
        ]
        return torch.cat(latents, dim=0)

    def train(self, df: pl.DataFrame) -> None:
        parameters = self._training_parameters()
        dimension = len(df["normalized_embed"][0])
        self.model = RQVAE(dim=dimension, **parameters)
        checkpoint_path = self.config.checkpoint_path
        self.checkpoint_loaded = False
        if checkpoint_path is not None:
            self.checkpoint_loaded = self._load_checkpoint(
                Path(checkpoint_path),
                dimension=dimension,
                item_count=df.height,
            )
        if self.checkpoint_loaded:
            self.tracker.log_metrics({"checkpoint.reused": 1})
            self.assignment_metrics = None
            return
        values = torch.from_numpy(self.model.converter(df)).to(self.model.device)
        self.model.pretrain_autoencoder(values, tracker=self.tracker)
        latent = self._encode_latents(values).cpu().numpy()
        initializer = RQKMeans(
            num_levels=self.config.num_levels,
            codebook_size=self.config.codebook_size,
        ).fit(latent)
        self.model.init_codebooks(initializer.codebooks)
        self.model.fit(values, tracker=self.tracker)
        if checkpoint_path is not None:
            self._save_checkpoint(
                Path(checkpoint_path),
                dimension=dimension,
                item_count=df.height,
            )
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
        return {
            "assignment": self.assignment_metrics,
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_path": self.config.checkpoint_path,
        }
