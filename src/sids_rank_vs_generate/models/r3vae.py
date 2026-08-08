from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def preference_discrimination_loss(
    codebook: torch.Tensor,
    *,
    temperature: float = 2.0,
    chunk_size: int = 2_048,
) -> torch.Tensor:
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [num_codes, embedding_dim]")
    num_codes = codebook.shape[0]
    if num_codes < 2:
        return codebook.sum() * 0.0
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    normalized = F.normalize(codebook, dim=-1)
    log_sum = normalized.new_tensor(float("-inf"))
    row_ids = torch.arange(num_codes, device=normalized.device)
    for start in range(0, num_codes, chunk_size):
        end = min(start + chunk_size, num_codes)
        similarities = normalized[start:end] @ normalized.T
        values = -temperature * (1.0 - similarities)
        diagonal = row_ids[start:end, None].eq(row_ids[None, :])
        values = values.masked_fill(diagonal, float("-inf"))
        log_sum = torch.logaddexp(
            log_sum,
            torch.logsumexp(values, dim=(0, 1)),
        )

    return log_sum - math.log(num_codes * (num_codes - 1))


@dataclass
class R3VAEOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    semantic_cohesion_loss: torch.Tensor
    preference_discrimination_loss: torch.Tensor
    reconstruction: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor
    codebook_usage: tuple[torch.Tensor, ...]


class RatingResidualQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        codebook_sizes: Sequence[int],
        *,
        top_k: int = 1,
        pd_temperature: float = 2.0,
        pd_chunk_size: int = 2_048,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        sizes = tuple(int(size) for size in codebook_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        if top_k <= 0 or any(top_k > size for size in sizes):
            raise ValueError("top_k must be positive and fit every codebook")

        self.embedding_dim = embedding_dim
        self.codebook_sizes = sizes
        self.top_k = top_k
        self.pd_temperature = pd_temperature
        self.pd_chunk_size = pd_chunk_size
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.empty(size, embedding_dim)) for size in sizes]
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for level, codebook in enumerate(self.codebooks):
            bound = 1.0 / (level + 1)
            nn.init.uniform_(codebook, -bound, bound)

    def forward(
        self,
        residual: torch.Tensor,
        *,
        initial_quantized: torch.Tensor,
        compute_losses: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        if residual.ndim != 2 or residual.shape[-1] != self.embedding_dim:
            raise ValueError(
                "residual must have shape [batch_size, embedding_dim]"
            )

        semantic_target = residual
        quantized = initial_quantized
        indices_by_level: list[torch.Tensor] = []
        cohesion_losses: list[torch.Tensor] = []
        discrimination_losses: list[torch.Tensor] = []
        usages: list[torch.Tensor] = []

        for codebook in self.codebooks:
            residual = F.normalize(residual, dim=-1)
            normalized_codebook = F.normalize(codebook, dim=-1)
            ratings = residual @ normalized_codebook.T
            values, indices = torch.topk(ratings, k=self.top_k, dim=-1)
            selected = normalized_codebook[indices]
            level_quantized = (selected * values.unsqueeze(-1)).sum(dim=1)

            residual = residual - level_quantized
            quantized = quantized + level_quantized
            hard_indices = indices[:, 0]
            indices_by_level.append(hard_indices)
            if compute_losses:
                cohesion_losses.append(
                    1.0
                    - F.cosine_similarity(
                        semantic_target,
                        quantized,
                        dim=-1,
                    ).mean()
                )
                discrimination_losses.append(
                    preference_discrimination_loss(
                        codebook,
                        temperature=self.pd_temperature,
                        chunk_size=self.pd_chunk_size,
                    )
                )
            usages.append(
                hard_indices.unique().numel() / codebook.shape[0]
                * torch.ones((), device=residual.device)
            )

        zero = quantized.sum() * 0.0
        return (
            quantized,
            torch.stack(indices_by_level, dim=1),
            torch.stack(cohesion_losses).mean() if compute_losses else zero,
            (
                torch.stack(discrimination_losses).mean()
                if compute_losses
                else zero
            ),
            tuple(usages),
        )


class R3VAE(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        codebook_sizes: Sequence[int],
        *,
        top_k: int = 1,
        reconstruction_loss_weight: float = 1.0,
        semantic_cohesion_loss_weight: float = 0.1,
        preference_discrimination_loss_weight: float = 0.05,
        pd_temperature: float = 2.0,
        pd_chunk_size: int = 2_048,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        for name, value in {
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "semantic_cohesion_loss_weight": semantic_cohesion_loss_weight,
            "preference_discrimination_loss_weight": (
                preference_discrimination_loss_weight
            ),
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.embedding_dim = embedding_dim
        self.reference_vector = nn.Parameter(torch.empty(1, embedding_dim))
        self.quantizer = RatingResidualQuantizer(
            embedding_dim,
            codebook_sizes,
            top_k=top_k,
            pd_temperature=pd_temperature,
            pd_chunk_size=pd_chunk_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.semantic_cohesion_loss_weight = semantic_cohesion_loss_weight
        self.preference_discrimination_loss_weight = (
            preference_discrimination_loss_weight
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.uniform_(self.reference_vector, -1.0, 1.0)
        self.quantizer.reset_parameters()
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def reference_projection(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "embeddings must have shape [batch_size, embedding_dim]"
            )
        reference = F.normalize(self.reference_vector, dim=-1)
        alpha = embeddings @ reference.T
        projection = alpha * reference
        return projection, embeddings - projection

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings.float(), dim=-1)
        reference_projection, residual = self.reference_projection(embeddings)
        _, indices, _, _, _ = self.quantizer(
            residual,
            initial_quantized=reference_projection,
            compute_losses=False,
        )
        return indices

    def forward(self, embeddings: torch.Tensor) -> R3VAEOutput:
        embeddings = F.normalize(embeddings.float(), dim=-1)
        reference_projection, residual = self.reference_projection(embeddings)
        (
            quantized,
            indices,
            cohesion_loss,
            discrimination_loss,
            usage,
        ) = self.quantizer(
            residual,
            initial_quantized=reference_projection,
        )
        reconstruction = self.decoder(quantized)
        reconstruction_loss = (
            1.0
            - F.cosine_similarity(
                reconstruction,
                embeddings,
                dim=-1,
            ).mean()
        )
        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.semantic_cohesion_loss_weight * cohesion_loss
            + self.preference_discrimination_loss_weight * discrimination_loss
        )
        return R3VAEOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            semantic_cohesion_loss=cohesion_loss,
            preference_discrimination_loss=discrimination_loss,
            reconstruction=reconstruction,
            quantized=quantized,
            indices=indices,
            codebook_usage=usage,
        )
