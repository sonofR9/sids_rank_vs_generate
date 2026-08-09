from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Literal, Sequence

import torch


AssignmentMethod = Literal["nearest", "ecm", "rrs"]


@dataclass(frozen=True)
class PureSemanticAllocation:
    codes: torch.Tensor
    metrics: dict[str, float | int | str]


def _validate_inputs(
    latents: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    top_k: int | Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    if latents.ndim != 2:
        raise ValueError("latents must have shape [num_items, latent_dim]")
    books = tuple(codebooks)
    if not books:
        raise ValueError("at least one codebook is required")
    if any(
        book.ndim != 2 or book.shape[1] != latents.shape[1]
        for book in books
    ):
        raise ValueError("all codebooks must have shape [num_codes, latent_dim]")
    if isinstance(top_k, int):
        top_ks = (top_k,) * len(books)
    else:
        top_ks = tuple(int(value) for value in top_k)
    if len(top_ks) != len(books):
        raise ValueError("top_k must contain one value per codebook")
    if any(value <= 0 for value in top_ks):
        raise ValueError("top_k values must be positive")
    if any(value > book.shape[0] for value, book in zip(top_ks, books)):
        raise ValueError("top_k must fit every codebook")
    return books, top_ks


def _squared_distances(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    return (
        values.square().sum(dim=-1, keepdim=True)
        - 2.0 * values @ codebook.T
        + codebook.square().sum(dim=-1).unsqueeze(0)
    )


@torch.no_grad()
def nearest_residual_codes(
    latents: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
) -> torch.Tensor:
    books, _ = _validate_inputs(latents, codebooks, 1)
    residual = latents
    codes: list[torch.Tensor] = []
    for codebook in books:
        indices = _squared_distances(residual, codebook).argmin(dim=-1)
        residual = residual - codebook[indices]
        codes.append(indices)
    return torch.stack(codes, dim=1)


def _codes_as_tuples(codes: torch.Tensor) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in row) for row in codes.cpu().tolist()]


def _collision_metrics(codes: torch.Tensor) -> dict[str, float | int]:
    tuples = _codes_as_tuples(codes)
    counts = Counter(tuples)
    item_count = len(tuples)
    singleton_items = sum(count for count in counts.values() if count == 1)
    unique_path_count = len(counts)
    icr = unique_path_count / item_count if item_count else 1.0
    return {
        "item_count": item_count,
        "unique_path_count": unique_path_count,
        "icr": icr,
        "collision_rate": 1.0 - icr,
        "singleton_count": singleton_items,
        "singleton_rate": singleton_items / item_count if item_count else 1.0,
        "colliding_item_count": item_count - singleton_items,
        "max_collision": max(counts.values(), default=0),
    }


def _residual_norms(
    latents: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    codes: torch.Tensor,
) -> torch.Tensor:
    residual = latents.clone()
    for level, codebook in enumerate(codebooks):
        residual = residual - codebook[codes[:, level]]
    return residual.norm(dim=-1)


def _ecm_candidates(
    latent: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    top_ks: Sequence[int],
) -> list[tuple[int, ...]]:
    residuals = latent.unsqueeze(0)
    codes = torch.empty((1, 0), dtype=torch.long, device=latent.device)
    cumulative_cost = latent.new_zeros(1)
    for codebook, level_k in zip(codebooks, top_ks, strict=True):
        distances = _squared_distances(residuals, codebook)
        indices = torch.topk(
            distances,
            k=level_k,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        selected = codebook[indices]
        next_residuals = residuals[:, None, :] - selected
        cumulative_cost = (
            cumulative_cost[:, None] + next_residuals.norm(dim=-1)
        ).reshape(-1)
        codes = torch.cat(
            (
                codes[:, None, :].expand(-1, level_k, -1),
                indices.unsqueeze(-1),
            ),
            dim=-1,
        ).reshape(-1, codes.shape[1] + 1)
        residuals = next_residuals.reshape(-1, latent.shape[0])
    order = cumulative_cost.argsort()
    return [
        tuple(int(value) for value in codes[index].tolist())
        for index in order
    ]


def _rrs_candidate(
    latent: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    top_ks: Sequence[int],
    used: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    def search(
        residual: torch.Tensor,
        level: int,
        prefix: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        if level == len(codebooks):
            return prefix if prefix not in used else None
        codebook = codebooks[level]
        distances = _squared_distances(residual.unsqueeze(0), codebook)[0]
        indices = torch.topk(
            distances,
            k=top_ks[level],
            largest=False,
            sorted=True,
        ).indices
        for index_tensor in indices:
            index = int(index_tensor.item())
            found = search(
                residual - codebook[index],
                level + 1,
                (*prefix, index),
            )
            if found is not None:
                return found
        return None

    return search(latent, 0, ())


@torch.no_grad()
def allocate_semantic_ids(
    latents: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    *,
    method: AssignmentMethod = "rrs",
    top_k: int | Sequence[int] = 4,
) -> PureSemanticAllocation:
    if method not in {"nearest", "ecm", "rrs"}:
        raise ValueError("method must be 'nearest', 'ecm', or 'rrs'")
    books, top_ks = _validate_inputs(latents, codebooks, top_k)
    nearest = nearest_residual_codes(latents, books)
    base_metrics = _collision_metrics(nearest)

    if method == "nearest":
        final_codes = nearest
    else:
        capacity = prod(book.shape[0] for book in books)
        if latents.shape[0] > capacity:
            raise RuntimeError(
                "unique semantic IDs are impossible: item count exceeds "
                f"code-space capacity ({latents.shape[0]} > {capacity})"
            )
        used: set[tuple[int, ...]] = set()
        assigned: list[tuple[int, ...]] = []
        nearest_tuples = _codes_as_tuples(nearest)
        for item_index, latent in enumerate(latents):
            if method == "ecm":
                selected = next(
                    (
                        candidate
                        for candidate in _ecm_candidates(latent, books, top_ks)
                        if candidate not in used
                    ),
                    None,
                )
            else:
                nearest_id = nearest_tuples[item_index]
                selected = (
                    nearest_id
                    if nearest_id not in used
                    else _rrs_candidate(latent, books, top_ks, used)
                )
            if selected is None:
                raise RuntimeError(
                    "PSI could not find an unused semantic ID for item "
                    f"{item_index}; increase psi_top_k or codebook capacity"
                )
            used.add(selected)
            assigned.append(selected)
        final_codes = torch.tensor(
            assigned,
            dtype=torch.long,
            device=latents.device,
        )

    final_metrics = _collision_metrics(final_codes)
    nearest_norms = _residual_norms(latents, books, nearest)
    final_norms = _residual_norms(latents, books, final_codes)
    metrics: dict[str, float | int | str] = {
        "method": method,
        "top_k": ",".join(str(value) for value in top_ks),
        **{f"base_{name}": value for name, value in base_metrics.items()},
        **{f"final_{name}": value for name, value in final_metrics.items()},
        "reassigned_item_count": int((final_codes != nearest).any(dim=1).sum()),
        "mean_nearest_residual_norm": float(nearest_norms.mean().item()),
        "mean_final_residual_norm": float(final_norms.mean().item()),
        "mean_residual_norm_increase": float(
            (final_norms - nearest_norms).mean().item()
        ),
    }
    return PureSemanticAllocation(codes=final_codes, metrics=metrics)
