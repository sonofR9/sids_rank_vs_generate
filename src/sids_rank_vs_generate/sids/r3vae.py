from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from ..models.r3vae import R3VAE
from ..reproducibility import seed_everything
from .base import ISidPipeline


@dataclass(frozen=True)
class R3VAEConfig:
    codebook_sizes: tuple[int, ...] = (256, 256, 256)
    top_k: int = 1
    epochs: int = 20
    batch_size: int = 1_024
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    reconstruction_loss_weight: float = 1.0
    semantic_cohesion_loss_weight: float = 0.1
    preference_discrimination_loss_weight: float = 0.05
    pd_temperature: float = 2.0
    pd_chunk_size: int = 2_048
    kmeans_init: bool = True
    kmeans_init_method: Literal["spherical", "rqkmeans"] = "spherical"
    kmeans_iterations: int = 20
    seed: int = 42
    device: str = "auto"
    feature_column: str | None = None

    def __post_init__(self) -> None:
        if not self.codebook_sizes or any(
            size <= 0 for size in self.codebook_sizes
        ):
            raise ValueError("codebook_sizes must contain positive integers")
        for name, value in {
            "top_k": self.top_k,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "pd_chunk_size": self.pd_chunk_size,
            "kmeans_iterations": self.kmeans_iterations,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if any(self.top_k > size for size in self.codebook_sizes):
            raise ValueError("top_k must fit every codebook")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.kmeans_init_method not in {"spherical", "rqkmeans"}:
            raise ValueError(
                "kmeans_init_method must be 'spherical' or 'rqkmeans'"
            )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _feature_column(frame: pl.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"feature column {requested!r} is missing")
        return requested
    for candidate in ("normalized_embed", "embed"):
        if candidate in frame.columns:
            return candidate
    candidates = [column for column in frame.columns if column != "item_id"]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "could not infer the embedding column; set R3VAEConfig.feature_column"
    )


def _embedding_tensor(frame: pl.DataFrame, column: str) -> torch.Tensor:
    if frame.is_empty():
        raise ValueError("item feature frame must not be empty")
    if frame[column].null_count():
        raise ValueError(f"embedding column {column!r} contains nulls")
    try:
        values = np.asarray(frame[column].to_list(), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"embedding column {column!r} must contain equal-length vectors"
        ) from error
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(
            f"embedding column {column!r} must contain non-empty vectors"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"embedding column {column!r} contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"embedding column {column!r} contains zero vectors")
    return torch.from_numpy(values)


@torch.no_grad()
def _spherical_kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    *,
    iterations: int,
    seed: int,
    assignment_chunk_size: int,
) -> torch.Tensor:
    samples = F.normalize(samples, dim=-1)
    sample_count = samples.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(sample_count, generator=generator)
    if num_clusters <= sample_count:
        initial_indices = order[:num_clusters]
    else:
        repeats = (num_clusters + sample_count - 1) // sample_count
        initial_indices = order.repeat(repeats)[:num_clusters]
    centers = samples[initial_indices.to(samples.device)].clone()

    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(
            num_clusters,
            dtype=samples.dtype,
            device=samples.device,
        )
        for start in range(0, sample_count, assignment_chunk_size):
            chunk = samples[start : start + assignment_chunk_size]
            assignments = (chunk @ centers.T).argmax(dim=1)
            sums.index_add_(0, assignments, chunk)
            counts.index_add_(
                0,
                assignments,
                torch.ones_like(assignments, dtype=samples.dtype),
            )
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        centers = F.normalize(centers, dim=-1)
    return centers


class R3VAESid(ISidPipeline):
    def __init__(self, config: R3VAEConfig | None = None) -> None:
        self.config = config or R3VAEConfig()
        self.model: R3VAE | None = None
        self.feature_column: str | None = None
        self.embedding_dim: int | None = None
        self.device = _resolve_device(self.config.device)
        self.training_history: list[dict[str, float]] = []

    @torch.no_grad()
    def _initialize_codebooks(self, embeddings: torch.Tensor) -> None:
        if self.model is None:
            raise RuntimeError("model has not been created")
        normalized = F.normalize(embeddings.to(self.device), dim=-1)
        _, residual = self.model.reference_projection(normalized)
        if self.config.kmeans_init_method == "rqkmeans":
            self._initialize_codebooks_with_rqkmeans(residual)
            return
        for level, codebook in enumerate(self.model.quantizer.codebooks):
            residual = F.normalize(residual, dim=-1)
            centers = _spherical_kmeans(
                residual,
                codebook.shape[0],
                iterations=self.config.kmeans_iterations,
                seed=self.config.seed + level,
                assignment_chunk_size=self.config.batch_size,
            )
            codebook.copy_(centers)
            ratings = residual @ centers.T
            values, indices = torch.topk(
                ratings,
                k=self.config.top_k,
                dim=-1,
            )
            level_quantized = (
                centers[indices] * values.unsqueeze(-1)
            ).sum(dim=1)
            residual = residual - level_quantized

    @torch.no_grad()
    def _initialize_codebooks_with_rqkmeans(
        self,
        residual: torch.Tensor,
    ) -> None:
        if self.model is None:
            raise RuntimeError("model has not been created")
        from .rq_kmeans import RQKMeans

        initializer = RQKMeans(
            codebook_sizes=self.config.codebook_sizes,
        ).fit(residual.detach().cpu().numpy())
        if len(initializer.codebooks) != len(self.model.quantizer.codebooks):
            raise RuntimeError(
                "RQKMeans returned an unexpected number of codebooks"
            )
        for parameter, centers in zip(
            self.model.quantizer.codebooks,
            initializer.codebooks,
            strict=True,
        ):
            values = torch.as_tensor(
                centers,
                dtype=parameter.dtype,
                device=parameter.device,
            )
            if values.shape != parameter.shape:
                raise RuntimeError(
                    "RQKMeans returned an unexpected codebook shape"
                )
            parameter.copy_(F.normalize(values, dim=-1))

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(self.config.seed)
        column = _feature_column(df, self.config.feature_column)
        embeddings = _embedding_tensor(df, column)
        self.feature_column = column
        self.embedding_dim = embeddings.shape[1]
        self.model = R3VAE(
            self.embedding_dim,
            self.config.codebook_sizes,
            top_k=self.config.top_k,
            reconstruction_loss_weight=(
                self.config.reconstruction_loss_weight
            ),
            semantic_cohesion_loss_weight=(
                self.config.semantic_cohesion_loss_weight
            ),
            preference_discrimination_loss_weight=(
                self.config.preference_discrimination_loss_weight
            ),
            pd_temperature=self.config.pd_temperature,
            pd_chunk_size=self.config.pd_chunk_size,
        ).to(self.device)
        if self.config.kmeans_init:
            self._initialize_codebooks(embeddings)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.training_history = []
        for epoch in range(self.config.epochs):
            self.model.train()
            totals = {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "semantic_cohesion_loss": 0.0,
                "preference_discrimination_loss": 0.0,
            }
            seen = 0
            permutation = torch.randperm(
                embeddings.shape[0],
                generator=generator,
            )
            for start in range(0, embeddings.shape[0], self.config.batch_size):
                batch_indices = permutation[start : start + self.config.batch_size]
                batch = embeddings[batch_indices].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch)
                if not torch.isfinite(output.loss):
                    raise RuntimeError("R3-VAE training produced a non-finite loss")
                output.loss.backward()
                optimizer.step()

                batch_size = batch.shape[0]
                seen += batch_size
                totals["loss"] += output.loss.item() * batch_size
                totals["reconstruction_loss"] += (
                    output.reconstruction_loss.item() * batch_size
                )
                totals["semantic_cohesion_loss"] += (
                    output.semantic_cohesion_loss.item() * batch_size
                )
                totals["preference_discrimination_loss"] += (
                    output.preference_discrimination_loss.item() * batch_size
                )
            epoch_metrics = {
                name: value / seen for name, value in totals.items()
            }
            self.training_history.append(epoch_metrics)
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.log_metrics(
                    {
                        f"train.{name}": value
                        for name, value in epoch_metrics.items()
                    },
                    step=epoch + 1,
                )
            print(
                f"R3-VAE epoch {epoch + 1}/{self.config.epochs}: "
                f"loss={epoch_metrics['loss']:.6f}, "
                f"rec={epoch_metrics['reconstruction_loss']:.6f}, "
                f"sc={epoch_metrics['semantic_cohesion_loss']:.6f}, "
                "pd="
                f"{epoch_metrics['preference_discrimination_loss']:.6f}",
                flush=True,
            )

    @torch.no_grad()
    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None or self.feature_column is None:
            raise RuntimeError("train() must be called before update()")
        embeddings = _embedding_tensor(df, self.feature_column)
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "embedding width differs from the data used during training"
            )
        self.model.eval()
        code_batches = []
        for start in range(0, embeddings.shape[0], self.config.batch_size):
            batch = embeddings[start : start + self.config.batch_size].to(
                self.device
            )
            code_batches.append(self.model.encode(batch).cpu())
        codes = torch.cat(code_batches, dim=0).numpy()
        result = df.select("item_id")
        for level in range(codes.shape[1]):
            result = result.with_columns(
                pl.Series(f"sid_{level}", codes[:, level].astype(np.int64))
            )
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "device": str(self.device),
            "feature_column": self.feature_column,
            "embedding_dim": self.embedding_dim,
            "epochs": list(self.training_history),
        }
