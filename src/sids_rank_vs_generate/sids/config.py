from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AssignmentMethod = Literal["nearest", "ecm", "rrs"]


@dataclass(frozen=True)
class RQKMeansConfig:
    num_levels: int | None = 3
    codebook_size: int | None = 512
    kmeans_batch_size: int = 16_384
    predict_batch_size: int = 65_536
    codebook_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.codebook_sizes is not None:
            if not self.codebook_sizes or any(
                size <= 0 for size in self.codebook_sizes
            ):
                raise ValueError("codebook_sizes must contain positive values")
        elif self.num_levels is None or self.codebook_size is None:
            raise ValueError("num_levels and codebook_size are required")


@dataclass(frozen=True)
class RQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 512
    batch_size: int = 256
    lr: float = 1e-3
    beta: float = 0.25
    num_workers: int = 4
    device: str | None = None
    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3
    max_epochs: int = 100
    patience: int = 5
    min_delta: float = 1e-5
    val_part: float = 0.05
    assignment_method: AssignmentMethod = "nearest"
    psi_top_k: int = 4

    def __post_init__(self) -> None:
        if self.assignment_method not in {"nearest", "ecm", "rrs"}:
            raise ValueError(
                "assignment_method must be 'nearest', 'ecm', or 'rrs'"
            )
        if self.psi_top_k <= 0 or self.psi_top_k > self.codebook_size:
            raise ValueError("psi_top_k must be positive and fit the codebook")


@dataclass(frozen=True)
class PlumRQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 2
    codebook_size: int = 512
    codebook_sizes: tuple[int, ...] = (512, 256)
    batch_size: int = 256
    num_epochs: int = 50
    lr: float = 5e-4
    beta: float = 0.25
    num_workers: int = 4
    device: str | None = None
    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3
    seed: int = 42

@dataclass(frozen=True)
class BalanceRQKMeansConfig:
    num_levels: int | None = 3
    codebook_size: int | None = 512
    codebook_sizes: tuple[int, ...] | None = None

    kmeans_batch_size: int = 65_536
    predict_batch_size: int = 16_384
    max_iter: int = 50
    device: str | None = None
    seed: int = 42


@dataclass(frozen=True)
class UserAwareRQKMeansConfig:
    """Настройки персонального RQ-KMeans"""

    num_levels: int | None = 3
    codebook_size: int | None = 256
    codebook_sizes: tuple[int, ...] | None = None
    embedding_column: str = "normalized_embed"
    user_profile_weight: float = 1.0
    kmeans_batch_size: int = 16_384
    predict_batch_size: int = 65_536

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

@dataclass(frozen=True)
class RQVAELetterConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 256
    batch_size: int = 1024
    num_epochs: int = 20
    lr: float = 5e-4
    beta: float = 0.25

    collaborative_embedding_dim: int = 128
    collaborative_embedding_column: str = "collaborative_embed"
    collaborative_weight: float = 2e-2

    diversity_clusters: int = 10
    diversity_weight: float = 1e-4

    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3
    num_workers: int = 4
    device: str | None = None

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if self.num_levels <= 0 or self.codebook_size <= 1:
            raise ValueError("num_levels must be positive and codebook_size must exceed one")
        if self.batch_size <= 0 or self.num_epochs <= 0 or self.num_workers < 0:
            raise ValueError("batch_size and num_epochs must be positive; num_workers must be non-negative")
        if self.lr <= 0 or self.pretrain_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.beta < 0 or self.collaborative_weight < 0 or self.diversity_weight < 0:
            raise ValueError("loss weights must be non-negative")
        # Each code needs another code from its cluster as a positive example.
        if self.diversity_clusters <= 0 or self.codebook_size < 2 * self.diversity_clusters:
            raise ValueError(
                "codebook_size must be at least twice diversity_clusters"
            )
        if self.collaborative_embedding_dim != self.latent_dim:
            raise ValueError(
                "collaborative_embedding_dim must equal latent_dim for the collaborative loss"
            )
        if not self.collaborative_embedding_column:
            raise ValueError("collaborative_embedding_column must not be empty")
