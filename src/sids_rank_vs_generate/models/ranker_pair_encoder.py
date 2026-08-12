from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class PairEncoderOutput(NamedTuple):
    logits: torch.Tensor
    embeddings: torch.Tensor


def build_pair_features(
    item_embeddings: torch.Tensor,
    user_profiles: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build the stable item-profile representation used by user-aware 2.0."""

    if item_embeddings.ndim != 2 or user_profiles.ndim != 2:
        raise ValueError("item embeddings and user profiles must be 2D")
    if item_embeddings.shape != user_profiles.shape:
        raise ValueError("item embeddings and user profiles must have equal shapes")
    if item_embeddings.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")
    if not item_embeddings.is_floating_point() or not user_profiles.is_floating_point():
        raise TypeError("item embeddings and user profiles must be floating point")

    items = F.normalize(item_embeddings.float(), dim=-1, eps=eps)
    profiles = F.normalize(user_profiles.float(), dim=-1, eps=eps)
    pair_features = torch.cat(
        (items, profiles, items * profiles, torch.abs(items - profiles)),
        dim=-1,
    )
    return F.normalize(pair_features, dim=-1, eps=eps)


class RankerPairEncoder(nn.Module):
    """Predict affinity and expose a normalized collaborative pair embedding."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 128),
        representation_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = int(embedding_dim)
        self.representation_dim = int(representation_dim)
        pair_dim = 4 * self.embedding_dim

        layers: list[nn.Module] = [nn.LayerNorm(pair_dim)]
        current_dim = pair_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.representation = nn.Linear(current_dim, self.representation_dim)
        self.classifier = nn.Linear(self.representation_dim, 1)

    def encode(
        self,
        item_embeddings: torch.Tensor,
        user_profiles: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = build_pair_features(item_embeddings, user_profiles)
        hidden = self.backbone(pair_features)
        representation = self.representation(hidden)
        return F.normalize(representation.float(), dim=-1, eps=1e-8)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        user_profiles: torch.Tensor,
    ) -> PairEncoderOutput:
        embeddings = self.encode(item_embeddings, user_profiles)
        logits = self.classifier(embeddings).squeeze(-1)
        return PairEncoderOutput(logits=logits, embeddings=embeddings)
