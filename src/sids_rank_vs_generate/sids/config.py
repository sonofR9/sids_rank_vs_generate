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
            if not self.codebook_sizes or any(size <= 0 for size in self.codebook_sizes):
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
