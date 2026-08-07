from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class PlumRQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 2048
    codebooks_sizes: tuple[int, ...] | None = None
    batch_size: int = 256
    num_epochs: int = 50
    lr: float = 5e-4
    beta: float = 0.25
    num_workers: int = 4
    device: str | None = None
    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3


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
