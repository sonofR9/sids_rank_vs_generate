from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F


LOGQ_MODES = frozenset({"none", "standard", "corrected"})


@dataclass
class RetrievalLossOutput:
    loss: torch.Tensor
    raw_logits: torch.Tensor
    corrected_logits: torch.Tensor
    labels: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def probabilities_to_log_q(
    probabilities: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert sampling probabilities to finite float32 log-probabilities."""

    if eps <= 0 or eps > 1:
        raise ValueError("logq eps must be in (0, 1]")
    values = torch.as_tensor(probabilities, dtype=torch.float32)
    values = torch.nan_to_num(values, nan=eps, posinf=1.0, neginf=eps)
    return values.clamp(min=eps, max=1.0).log()


def counts_to_log_q(
    counts: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert item occurrence counts to safe sampling log-probabilities."""

    if eps <= 0 or eps > 1:
        raise ValueError("logq eps must be in (0, 1]")
    values = torch.as_tensor(counts).to(dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("counts must be a non-empty 1D tensor")
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values.clamp_min(0.0)
    total = values.sum()
    if total <= 0:
        return torch.full_like(values, math.log(eps))
    return probabilities_to_log_q(values / total, eps=eps)


def sanitize_log_q(
    log_q: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return safe logQ values and the fraction that required clamping."""

    if eps <= 0 or eps > 1:
        raise ValueError("logq eps must be in (0, 1]")
    values = torch.as_tensor(log_q).to(dtype=torch.float32)
    lower = math.log(eps)
    invalid = ~torch.isfinite(values) | (values < lower) | (values > 0)
    values = torch.nan_to_num(values, nan=lower, neginf=lower, posinf=0.0)
    values = values.clamp(min=lower, max=0.0)
    return values, invalid.float().mean()


def apply_log_q_to_logits(
    logits: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    """Apply a broadcastable LogQ correction in stable float32 precision."""

    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    correction = torch.as_tensor(log_q, device=logits.device, dtype=torch.float32)
    try:
        torch.broadcast_shapes(tuple(logits.shape), tuple(correction.shape))
    except RuntimeError as error:
        raise ValueError("log_q is not broadcastable to logits") from error
    return logits.float() - correction


def apply_logq_correction(
    logits: torch.Tensor,
    log_q: torch.Tensor,
    labels: torch.Tensor,
    *,
    preserve_positive: bool,
) -> torch.Tensor:
    """Subtract per-document logQ, optionally restoring deterministic positives."""

    if logits.ndim != 2:
        raise ValueError("retrieval logits must be 2D")
    if log_q.ndim != 1 or log_q.numel() != logits.size(1):
        raise ValueError("log_q must contain one value per document column")
    if labels.shape != (logits.size(0),):
        raise ValueError("labels must contain one document column per query")
    if torch.any((labels < 0) | (labels >= logits.size(1))):
        raise ValueError("retrieval labels are out of range")

    # The model score is already temperature-scaled. Sampled-softmax importance
    # correction is therefore score / temperature - log Q, not
    # (score - log Q) / temperature: logQ lives in softmax-logit units.
    corrected = apply_log_q_to_logits(logits, log_q.unsqueeze(0))
    if preserve_positive:
        positive = logits.gather(1, labels.unsqueeze(1))
        corrected.scatter_(1, labels.unsqueeze(1), positive)
    return corrected


def corrected_logq_loss(
    raw_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refined LogQ objective and its stop-gradient positive probability."""

    if raw_logits.shape != corrected_logits.shape or raw_logits.ndim != 2:
        raise ValueError("raw and corrected logits must have the same 2D shape")
    if labels.shape != (raw_logits.size(0),):
        raise ValueError("labels must contain one document column per query")
    if corrected_logits.size(1) == 1:
        probability = raw_logits.new_ones(raw_logits.size(0)).detach()
        return raw_logits.sum() * 0.0, probability

    positive_logits = raw_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(corrected_logits, dtype=torch.bool)
    negative_mask.scatter_(1, labels.unsqueeze(1), False)
    negative_logits = corrected_logits.masked_fill(~negative_mask, -torch.inf)
    log_negative_sum = torch.logsumexp(negative_logits, dim=1)
    log_negative_mean = log_negative_sum - math.log(corrected_logits.size(1) - 1)
    probability = torch.sigmoid(positive_logits - log_negative_mean)
    # Corrected LogQ defines w=sg(1-P_positive). Detaching is essential: w is
    # a gradient multiplier; differentiating through it adds an extra term not
    # present in the importance-sampling derivation.
    weight = (1.0 - probability).detach()
    return (weight * (log_negative_sum - positive_logits)).mean(), probability.detach()


def concatenate_gathered_candidates(
    gathered_items: tuple[torch.Tensor, ...] | list[torch.Tensor],
    gathered_log_q: tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    sizes: tuple[int, ...] | list[int],
    *,
    rank: int,
    local_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Unpad rank-ordered gathers and build labels for this rank's positives."""

    if len(gathered_items) != len(sizes) or not 0 <= rank < len(sizes):
        raise ValueError("gathered candidate metadata is inconsistent")
    if sizes[rank] != local_batch_size:
        raise ValueError("local batch size does not match gathered rank size")
    items = torch.cat(
        [rank_items[:size] for rank_items, size in zip(gathered_items, sizes)],
        dim=0,
    )
    global_log_q = None
    if gathered_log_q is not None:
        if len(gathered_log_q) != len(sizes):
            raise ValueError("gathered log_q metadata is inconsistent")
        global_log_q = torch.cat(
            [rank_log_q[:size] for rank_log_q, size in zip(gathered_log_q, sizes)],
            dim=0,
        )
    offset = sum(sizes[:rank])
    labels = torch.arange(
        local_batch_size,
        device=items.device,
        dtype=torch.long,
    ) + offset
    return items, global_log_q, labels


def _distributed_candidates(
    item_embeddings: torch.Tensor,
    log_q: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    local_size = item_embeddings.size(0)
    if not dist.is_available() or not dist.is_initialized():
        labels = torch.arange(local_size, device=item_embeddings.device)
        return item_embeddings, log_q, labels

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    size_tensor = torch.tensor(
        [local_size], device=item_embeddings.device, dtype=torch.long
    )
    gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, size_tensor)
    sizes = [int(value.item()) for value in gathered_sizes]
    max_size = max(sizes)

    item_padding = (0, 0, 0, max_size - local_size)
    padded_items = F.pad(item_embeddings, item_padding)
    # torch.distributed.nn.all_gather is differentiable: gradients produced when
    # a local query uses a remote item are reduced back to that item's owner.
    gathered_items = dist_nn.all_gather(padded_items)

    gathered_log_q = None
    if log_q is not None:
        padded_log_q = F.pad(log_q, (0, max_size - local_size))
        gathered_log_q = [torch.empty_like(padded_log_q) for _ in range(world_size)]
        dist.all_gather(gathered_log_q, padded_log_q)

    return concatenate_gathered_candidates(
        gathered_items,
        gathered_log_q,
        sizes,
        rank=rank,
        local_batch_size=local_size,
    )


class InBatchRetrievalLoss(nn.Module):
    """In-batch sampled-softmax with standard or refined LogQ correction."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        logq_mode: str = "none",
        eps: float = 1e-12,
        debug: bool = False,
        gather_distributed: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("ranking temperature must be positive")
        if logq_mode not in LOGQ_MODES:
            raise ValueError(f"unknown logq mode: {logq_mode!r}")
        if eps <= 0 or eps > 1:
            raise ValueError("logq eps must be in (0, 1]")
        self.temperature = float(temperature)
        self.logq_mode = logq_mode
        self.eps = float(eps)
        self.debug = bool(debug)
        self.gather_distributed = bool(gather_distributed)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        log_q: torch.Tensor | None = None,
    ) -> RetrievalLossOutput:
        if query_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ValueError("query and item embeddings must be 2D")
        if query_embeddings.shape != item_embeddings.shape:
            raise ValueError("local query and item embedding shapes must match")
        if log_q is not None and log_q.shape != (item_embeddings.size(0),):
            raise ValueError("log_q must contain one value per local item")
        if self.logq_mode != "none" and log_q is None:
            raise ValueError(f"log_q is required for logq_mode={self.logq_mode!r}")

        local_log_q = None if log_q is None else log_q.to(item_embeddings.device)
        if self.gather_distributed:
            global_items, global_log_q, labels = _distributed_candidates(
                item_embeddings, local_log_q
            )
        else:
            global_items = item_embeddings
            global_log_q = local_log_q
            labels = torch.arange(item_embeddings.size(0), device=item_embeddings.device)

        loss_scale = 1.0
        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            # DDP averages rank gradients. This factor makes that average equal
            # a global per-query mean even if the final local batches are uneven.
            loss_scale = (
                dist.get_world_size()
                * item_embeddings.size(0)
                / global_items.size(0)
            )

        queries = F.normalize(query_embeddings, dim=-1)
        items = F.normalize(global_items, dim=-1)
        # Carry the numerically sensitive softmax/correction path in fp32 even
        # when encoders run under fp16/bf16 autocast.
        raw_logits = (queries @ items.T).float() / self.temperature
        diagnostics: dict[str, torch.Tensor] = {}

        if self.logq_mode == "none":
            loss = F.cross_entropy(raw_logits, labels) * loss_scale
            return RetrievalLossOutput(loss, raw_logits, raw_logits, labels, diagnostics)

        assert global_log_q is not None
        safe_log_q, clamp_fraction = sanitize_log_q(global_log_q, eps=self.eps)
        corrected_logits = apply_logq_correction(
            raw_logits,
            safe_log_q,
            labels,
            preserve_positive=self.logq_mode == "corrected",
        )

        if self.logq_mode == "standard":
            loss = F.cross_entropy(corrected_logits, labels)
        else:
            loss, _positive_probability = corrected_logq_loss(
                raw_logits, corrected_logits, labels
            )
        loss = loss * loss_scale

        if self.debug:
            with torch.no_grad():
                positive_before = raw_logits.gather(
                    1, labels.unsqueeze(1)
                ).squeeze(1)
                positive_after = corrected_logits.gather(
                    1, labels.unsqueeze(1)
                ).squeeze(1)
                negative_mask = torch.ones_like(raw_logits, dtype=torch.bool)
                negative_mask.scatter_(1, labels.unsqueeze(1), False)
                negative_before = raw_logits[negative_mask]
                negative_after = corrected_logits[negative_mask]
                diagnostics = {
                    "log_q_mean": safe_log_q.mean(),
                    "log_q_min": safe_log_q.min(),
                    "log_q_max": safe_log_q.max(),
                    "correction_magnitude_mean": (
                        (negative_after - negative_before).abs().mean()
                        if negative_before.numel()
                        else raw_logits.new_zeros(())
                    ),
                    "positive_logit_before": positive_before.mean(),
                    "positive_logit_after": positive_after.mean(),
                    "negative_logit_before": (
                        negative_before.mean()
                        if negative_before.numel()
                        else raw_logits.new_zeros(())
                    ),
                    "negative_logit_after": (
                        negative_after.mean()
                        if negative_after.numel()
                        else raw_logits.new_zeros(())
                    ),
                    "loss_without_logq": (
                        F.cross_entropy(raw_logits, labels) * loss_scale
                    ),
                    "loss_with_logq": loss,
                    "clamp_fraction": clamp_fraction,
                }

        return RetrievalLossOutput(
            loss,
            raw_logits,
            corrected_logits,
            labels,
            diagnostics,
        )
