from __future__ import annotations

from dataclasses import dataclass, field
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
class CollaborativeAlignmentConfig:
    """Pair mining and contrastive alignment settings for OneRec-style SIDs."""

    embedding_column: str = "normalized_embed"
    history_window: int = 3
    min_pair_support: int = 2
    pairs_per_item: int = 20
    max_pairs: int | None = 1_000_000
    hidden_dim: int = 512
    batch_size: int = 1024
    inference_batch_size: int = 65_536
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    content_preservation_weight: float = 0.1
    device: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.embedding_column:
            raise ValueError("embedding_column must not be empty")
        positive_integers = {
            "history_window": self.history_window,
            "min_pair_support": self.min_pair_support,
            "pairs_per_item": self.pairs_per_item,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "epochs": self.epochs,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive when provided")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.content_preservation_weight < 0:
            raise ValueError("content_preservation_weight must be non-negative")


@dataclass(frozen=True)
class CollaborativeAwareRQKMeansConfig(RQKMeansConfig):
    """RQ-KMeans settings plus collaborative representation alignment."""

    alignment: CollaborativeAlignmentConfig = field(
        default_factory=CollaborativeAlignmentConfig
    )


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
class CollaborativeAwareRQVAEConfig(RQVAEConfig):
    """RQ-VAE settings plus train-only collaborative pair alignment."""

    alignment: CollaborativeAlignmentConfig = field(
        default_factory=CollaborativeAlignmentConfig
    )


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
