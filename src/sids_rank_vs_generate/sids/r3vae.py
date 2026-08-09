from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
    mixed_precision: bool = True
    checkpoint_path: Path | None = None
    reuse_checkpoint: bool = True

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
    series = frame[column]
    try:
        first = series[0]
        width = len(first)
        if isinstance(series.dtype, pl.Array):
            values = series.cast(
                pl.Array(pl.Float32, series.dtype.size)
            ).to_numpy()
        elif isinstance(series.dtype, pl.List):
            values = (
                series.list.to_array(width)
                .cast(pl.Array(pl.Float32, width))
                .to_numpy()
            )
        else:
            values = np.asarray(series.to_list(), dtype=np.float32)
        values = np.array(values, dtype=np.float32, copy=True, order="C")
    except (AttributeError, TypeError, ValueError) as error:
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


def _cuda_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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
    """Train R3-VAE on item vectors and append one SID token per codebook."""

    def __init__(self, config: R3VAEConfig | None = None) -> None:
        self.config = config or R3VAEConfig()
        self.model: R3VAE | None = None
        self.feature_column: str | None = None
        self.embedding_dim: int | None = None
        self.device = _resolve_device(self.config.device)
        self.training_history: list[dict[str, float]] = []
        self.checkpoint_loaded = False

    def _checkpoint_signature(self) -> dict[str, Any]:
        signature = asdict(self.config)
        for name in (
            "checkpoint_path",
            "reuse_checkpoint",
            "device",
        ):
            signature.pop(name)
        return signature

    def _load_checkpoint(
        self,
        path: Path,
        *,
        embedding_dim: int,
        item_count: int,
    ) -> bool:
        if not self.config.reuse_checkpoint or not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch 2.2 compatibility.
            payload = torch.load(path, map_location=self.device)
        expected = {
            "format_version": 1,
            "embedding_dim": embedding_dim,
            "item_count": item_count,
            "training_config": self._checkpoint_signature(),
        }
        actual = {name: payload.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"R3-VAE checkpoint {path} does not match the current dataset/config"
            )
        if self.model is None:
            raise RuntimeError("model has not been created")
        self.model.load_state_dict(payload["model_state_dict"])
        self.training_history = list(payload.get("training_history", []))
        return True

    def _save_checkpoint(
        self,
        path: Path,
        *,
        embedding_dim: int,
        item_count: int,
    ) -> None:
        if self.model is None:
            raise RuntimeError("model has not been created")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "embedding_dim": embedding_dim,
                "item_count": item_count,
                "training_config": self._checkpoint_signature(),
                "model_state_dict": self.model.state_dict(),
                "training_history": self.training_history,
            },
            path,
        )

    @torch.inference_mode()
    def _initialize_codebooks(self, embeddings: torch.Tensor) -> None:
        if self.model is None:
            raise RuntimeError("model has not been created")
        normalized = F.normalize(embeddings.to(self.device), dim=-1)
        _, residual = self.model.reference_projection(normalized)
        if self.config.kmeans_init_method == "rqkmeans":
            from .rq_kmeans import RQKMeans

            initializer = RQKMeans(
                codebook_sizes=self.config.codebook_sizes,
            ).fit(residual.detach().cpu().numpy())
            for codebook, centers in zip(
                self.model.quantizer.codebooks,
                initializer.codebooks,
                strict=True,
            ):
                normalized_centers = F.normalize(
                    torch.as_tensor(
                        centers,
                        dtype=codebook.dtype,
                        device=self.device,
                    ),
                    dim=-1,
                )
                codebook.copy_(normalized_centers)
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
        checkpoint_path = self.config.checkpoint_path
        self.checkpoint_loaded = False
        if checkpoint_path is not None:
            self.checkpoint_loaded = self._load_checkpoint(
                Path(checkpoint_path),
                embedding_dim=self.embedding_dim,
                item_count=df.height,
            )
        if self.checkpoint_loaded:
            self.tracker.log_metrics({"checkpoint.reused": 1})
            return
        if self.config.kmeans_init:
            self._initialize_codebooks(embeddings)

        embeddings_on_device = embeddings.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=self.device.type == "cuda",
        )
        autocast_dtype = _cuda_autocast_dtype(self.device)
        use_autocast = self.config.mixed_precision and self.device.type == "cuda"
        grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_autocast and autocast_dtype == torch.float16,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.training_history = []
        for epoch in range(self.config.epochs):
            self.model.train()
            metric_names = (
                "loss",
                "reconstruction_loss",
                "semantic_cohesion_loss",
                "preference_discrimination_loss",
            )
            totals = torch.zeros(4, dtype=torch.float32, device=self.device)
            seen = 0
            permutation = torch.randperm(
                embeddings.shape[0],
                generator=generator,
            ).to(self.device)
            for start in range(0, embeddings.shape[0], self.config.batch_size):
                batch_indices = permutation[start : start + self.config.batch_size]
                batch = embeddings_on_device[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=autocast_dtype,
                    enabled=use_autocast,
                ):
                    output = self.model(batch, compute_usage=False)
                grad_scaler.scale(output.loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                batch_size = batch.shape[0]
                seen += batch_size
                totals.add_(
                    torch.stack(
                        (
                            output.loss,
                            output.reconstruction_loss,
                            output.semantic_cohesion_loss,
                            output.preference_discrimination_loss,
                        )
                    ).detach().float(),
                    alpha=batch_size,
                )
            averages = (totals / seen).cpu().tolist()
            if not all(np.isfinite(value) for value in averages):
                raise RuntimeError("R3-VAE training produced a non-finite loss")
            epoch_metrics = dict(zip(metric_names, averages, strict=True))
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
        if checkpoint_path is not None:
            self._save_checkpoint(
                Path(checkpoint_path),
                embedding_dim=self.embedding_dim,
                item_count=df.height,
            )

    @torch.inference_mode()
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
                self.device,
                non_blocking=True,
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
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_path": self.config.checkpoint_path,
        }
