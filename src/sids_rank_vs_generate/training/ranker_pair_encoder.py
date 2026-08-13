from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..evaluation.ranking_metrics import average_precision, roc_auc
from ..models.ranker_pair_encoder import RankerPairEncoder
from ..reproducibility import seed_everything
from .utils import resolve_device


Precision = Literal["fp32", "fp16"]
PositiveWeight = float | Literal["auto"] | None
TablePlacement = Literal["auto", "cpu", "device"]


@dataclass(frozen=True)
class RankerPairEncoderTrainingConfig:
    batch_size: int = 8_192
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    pos_weight: PositiveWeight = "auto"
    precision: Precision = "fp32"
    device: str = "auto"
    table_placement: TablePlacement = "auto"
    max_device_table_fraction: float = 0.5
    seed: int = 42
    num_workers: int = 0
    reset_model_parameters: bool = True
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be 'fp32' or 'fp16'")
        if self.table_placement not in {"auto", "cpu", "device"}:
            raise ValueError("table_placement must be 'auto', 'cpu', or 'device'")
        if not 0.0 < self.max_device_table_fraction < 1.0:
            raise ValueError("max_device_table_fraction must be in (0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if isinstance(self.pos_weight, str):
            if self.pos_weight != "auto":
                raise ValueError("pos_weight must be positive, 'auto', or None")
        elif self.pos_weight is not None:
            if isinstance(self.pos_weight, bool) or float(self.pos_weight) <= 0:
                raise ValueError("pos_weight must be positive, 'auto', or None")


@dataclass(frozen=True)
class RankerPairEncoderTrainingResult:
    device: str
    table_device: str
    precision: Precision
    examples: int
    positives: int
    negatives: int
    pos_weight: float | None
    epoch_losses: tuple[float, ...]
    metrics: dict[str, float | int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_finite_in_chunks(tensor: torch.Tensor, name: str) -> None:
    row_count = tensor.shape[0]
    for offset in range(0, row_count, 65_536):
        if not torch.isfinite(tensor[offset : offset + 65_536]).all():
            raise ValueError(f"{name} must contain only finite values")


def _prepare_embedding_table(
    values: torch.Tensor,
    name: str,
) -> torch.Tensor:
    table = torch.as_tensor(values).detach().cpu().float().contiguous()
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor")
    _check_finite_in_chunks(table, name)
    return table


def _prepare_indices(
    values: torch.Tensor,
    *,
    name: str,
    upper_bound: int,
) -> torch.Tensor:
    indices = torch.as_tensor(values).detach().cpu().reshape(-1)
    if indices.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if indices.is_floating_point() or indices.dtype == torch.bool:
        raise TypeError(f"{name} must contain integer indices")
    indices = indices.long().contiguous()
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= upper_bound:
        raise ValueError(f"{name} contains an index outside its table")
    return indices


def _prepare_targets(values: torch.Tensor, expected_size: int) -> torch.Tensor:
    targets = torch.as_tensor(values).detach().cpu().float().reshape(-1).contiguous()
    if targets.numel() != expected_size:
        raise ValueError("targets must contain one label per indexed pair")
    for offset in range(0, targets.numel(), 65_536):
        part = targets[offset : offset + 65_536]
        if not torch.isfinite(part).all():
            raise ValueError("targets must contain only finite values")
        if not torch.all((part == 0) | (part == 1)):
            raise ValueError("targets must contain only binary labels")
    return targets


def _prepare_indexed_pairs(
    item_embedding_table: torch.Tensor,
    user_profile_table: torch.Tensor,
    item_indices: torch.Tensor,
    profile_indices: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    items = _prepare_embedding_table(item_embedding_table, "item embedding table")
    profiles = _prepare_embedding_table(user_profile_table, "user profile table")
    if items.shape[1] != profiles.shape[1]:
        raise ValueError("item and profile tables must have equal embedding widths")
    item_rows = _prepare_indices(
        item_indices,
        name="item indices",
        upper_bound=items.shape[0],
    )
    profile_rows = _prepare_indices(
        profile_indices,
        name="profile indices",
        upper_bound=profiles.shape[0],
    )
    if item_rows.numel() != profile_rows.numel():
        raise ValueError("item and profile indices must have equal lengths")
    labels = (
        None
        if targets is None
        else _prepare_targets(targets, expected_size=item_rows.numel())
    )
    return items, profiles, item_rows, profile_rows, labels


def _validate_model_dimension(
    model: RankerPairEncoder,
    item_table: torch.Tensor,
) -> None:
    if model.embedding_dim != item_table.shape[1]:
        raise ValueError(
            "model embedding dimension does not match the embedding tables"
        )


def _resolve_precision(precision: Precision, device: torch.device) -> bool:
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be 'fp32' or 'fp16'")
    use_fp16 = precision == "fp16"
    if use_fp16 and device.type != "cuda":
        raise ValueError("fp16 precision is supported only on CUDA")
    return use_fp16


def _resolve_requested_device(device: str | torch.device) -> torch.device:
    return resolve_device(str(device))


def _make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


def _reset_parameters(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def _resolve_pos_weight(
    targets: torch.Tensor,
    configured: PositiveWeight,
) -> tuple[float | None, int, int]:
    positives = int(targets.sum().item())
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        raise ValueError("fit targets must contain both classes")
    if configured is None:
        return None, positives, negatives
    if configured == "auto":
        return negatives / positives, positives, negatives
    return float(configured), positives, negatives


def _place_embedding_tables(
    item_table: torch.Tensor,
    profile_table: torch.Tensor,
    *,
    device: torch.device,
    placement: TablePlacement,
    max_device_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    use_device = placement == "device"
    if placement == "auto" and device.type == "cuda":
        table_bytes = (
            item_table.numel() * item_table.element_size()
            + profile_table.numel() * profile_table.element_size()
        )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        use_device = table_bytes <= int(free_bytes * max_device_fraction)
    if device.type == "cpu":
        use_device = False
    if not use_device:
        return item_table, profile_table, "cpu"
    return item_table.to(device), profile_table.to(device), str(device)


def _gather_pair_batch(
    item_table: torch.Tensor,
    profile_table: torch.Tensor,
    item_indices: torch.Tensor,
    profile_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if item_table.device.type == "cpu":
        items = item_table[item_indices].to(device)
        profiles = profile_table[profile_indices].to(device)
        return items, profiles
    item_indices = item_indices.to(device, non_blocking=True)
    profile_indices = profile_indices.to(device, non_blocking=True)
    return item_table[item_indices], profile_table[profile_indices]


@torch.inference_mode()
def encode_ranker_pairs_indexed(
    model: RankerPairEncoder,
    item_embedding_table: torch.Tensor,
    user_profile_table: torch.Tensor,
    item_indices: torch.Tensor,
    profile_indices: torch.Tensor,
    *,
    batch_size: int = 65_536,
    device: str | torch.device = "auto",
    precision: Precision = "fp32",
    table_placement: TablePlacement = "auto",
    max_device_table_fraction: float = 0.5,
) -> torch.Tensor:
    """Encode indexed pairs in batches without materializing their input vectors."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if table_placement not in {"auto", "cpu", "device"}:
        raise ValueError("table_placement must be 'auto', 'cpu', or 'device'")
    if not 0.0 < max_device_table_fraction < 1.0:
        raise ValueError("max_device_table_fraction must be in (0, 1)")
    items, profiles, item_rows, profile_rows, _ = _prepare_indexed_pairs(
        item_embedding_table,
        user_profile_table,
        item_indices,
        profile_indices,
    )
    _validate_model_dimension(model, items)
    resolved_device = _resolve_requested_device(device)
    use_fp16 = _resolve_precision(precision, resolved_device)
    model.to(resolved_device)
    items, profiles, _ = _place_embedding_tables(
        items,
        profiles,
        device=resolved_device,
        placement=table_placement,
        max_device_fraction=max_device_table_fraction,
    )

    was_training = model.training
    model.eval()
    output = torch.empty(
        (item_rows.numel(), model.representation_dim),
        dtype=torch.float32,
    )
    try:
        for offset in range(0, item_rows.numel(), batch_size):
            stop = min(offset + batch_size, item_rows.numel())
            item_batch, profile_batch = _gather_pair_batch(
                items,
                profiles,
                item_rows[offset:stop],
                profile_rows[offset:stop],
                resolved_device,
            )
            with _autocast(resolved_device, use_fp16):
                embeddings = model.encode(item_batch, profile_batch)
            output[offset:stop].copy_(embeddings.float().cpu())
    finally:
        model.train(was_training)
    return output.detach()


def encode_ranker_pairs(
    model: RankerPairEncoder,
    item_embeddings: torch.Tensor,
    user_profiles: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Convenience wrapper for already materialized dense pairs."""

    items = torch.as_tensor(item_embeddings)
    profiles = torch.as_tensor(user_profiles)
    if items.ndim != 2 or profiles.ndim != 2 or items.shape != profiles.shape:
        raise ValueError("item embeddings and user profiles must have equal 2D shapes")
    rows = torch.arange(items.shape[0])
    return encode_ranker_pairs_indexed(
        model,
        items,
        profiles,
        rows,
        rows,
        **kwargs,
    )


@torch.inference_mode()
def _fit_metrics(
    model: RankerPairEncoder,
    item_table: torch.Tensor,
    profile_table: torch.Tensor,
    item_indices: torch.Tensor,
    profile_indices: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
    pos_weight: float | None,
) -> dict[str, float | int]:
    model.eval()
    logits = torch.empty(targets.numel(), dtype=torch.float32)
    for offset in range(0, targets.numel(), batch_size):
        stop = min(offset + batch_size, targets.numel())
        item_batch, profile_batch = _gather_pair_batch(
            item_table,
            profile_table,
            item_indices[offset:stop],
            profile_indices[offset:stop],
            device,
        )
        with _autocast(device, use_fp16):
            batch_logits = model(item_batch, profile_batch).logits
        logits[offset:stop].copy_(batch_logits.float().cpu())

    probabilities = torch.sigmoid(logits)
    predictions = logits >= 0
    labels = targets.bool()
    true_positive = int((predictions & labels).sum().item())
    true_negative = int((~predictions & ~labels).sum().item())
    false_positive = int((predictions & ~labels).sum().item())
    false_negative = int((~predictions & labels).sum().item())
    positives = true_positive + false_negative
    negatives = true_negative + false_positive

    weight_tensor = None if pos_weight is None else torch.tensor(pos_weight)
    weighted_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight_tensor,
    )
    logloss = F.binary_cross_entropy_with_logits(logits, targets)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / positives
    specificity = true_negative / negatives
    return {
        "loss": float(weighted_loss.item()),
        "logloss": float(logloss.item()),
        "roc_auc": float(roc_auc(targets, probabilities).item()),
        "pr_auc": float(average_precision(targets, probabilities).item()),
        "accuracy": (true_positive + true_negative) / targets.numel(),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "positive_rate": positives / targets.numel(),
        "predicted_positive_rate": int(predictions.sum().item()) / targets.numel(),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def fit_ranker_pair_encoder(
    model: RankerPairEncoder,
    item_embedding_table: torch.Tensor,
    user_profile_table: torch.Tensor,
    item_indices: torch.Tensor,
    profile_indices: torch.Tensor,
    fit_targets: torch.Tensor,
    config: RankerPairEncoderTrainingConfig | None = None,
) -> RankerPairEncoderTrainingResult:
    """Fit on indexed behavior rows from the fit split only."""

    config = config or RankerPairEncoderTrainingConfig()
    items, profiles, item_rows, profile_rows, targets = _prepare_indexed_pairs(
        item_embedding_table,
        user_profile_table,
        item_indices,
        profile_indices,
        fit_targets,
    )
    assert targets is not None
    _validate_model_dimension(model, items)
    pos_weight, positives, negatives = _resolve_pos_weight(
        targets,
        config.pos_weight,
    )

    seed_everything(config.seed)
    device = resolve_device(config.device)
    use_fp16 = _resolve_precision(config.precision, device)
    if config.reset_model_parameters:
        model.apply(_reset_parameters)
    model.to(device)
    items, profiles, table_device = _place_embedding_tables(
        items,
        profiles,
        device=device,
        placement=config.table_placement,
        max_device_fraction=config.max_device_table_fraction,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(
            None
            if pos_weight is None
            else torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
    )
    grad_scaler = _make_grad_scaler(use_fp16)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(item_rows, profile_rows, targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    epoch_losses: list[float] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(
            loader,
            desc=f"pair encoder epoch {epoch}/{config.epochs}",
            disable=not config.show_progress,
        )
        for item_batch_rows, profile_batch_rows, target_batch in progress:
            item_batch, profile_batch = _gather_pair_batch(
                items,
                profiles,
                item_batch_rows,
                profile_batch_rows,
                device,
            )
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_fp16):
                logits = model(item_batch, profile_batch).logits
                loss = criterion(logits, target_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("pair encoder produced a non-finite loss")
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            batch_examples = target_batch.numel()
            loss_sum += float(loss.detach().item()) * batch_examples
            example_count += batch_examples
            progress.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = loss_sum / example_count
        if not math.isfinite(epoch_loss):
            raise RuntimeError("pair encoder produced a non-finite epoch loss")
        epoch_losses.append(epoch_loss)

    metrics = _fit_metrics(
        model,
        items,
        profiles,
        item_rows,
        profile_rows,
        targets,
        batch_size=config.batch_size,
        device=device,
        use_fp16=use_fp16,
        pos_weight=pos_weight,
    )
    return RankerPairEncoderTrainingResult(
        device=str(device),
        table_device=table_device,
        precision=config.precision,
        examples=targets.numel(),
        positives=positives,
        negatives=negatives,
        pos_weight=pos_weight,
        epoch_losses=tuple(epoch_losses),
        metrics=metrics,
    )


def fit_ranker_pair_encoder_dense(
    model: RankerPairEncoder,
    fit_item_embeddings: torch.Tensor,
    fit_user_profiles: torch.Tensor,
    fit_targets: torch.Tensor,
    config: RankerPairEncoderTrainingConfig | None = None,
) -> RankerPairEncoderTrainingResult:
    """Convenience wrapper for small datasets with materialized pair vectors."""

    items = torch.as_tensor(fit_item_embeddings)
    profiles = torch.as_tensor(fit_user_profiles)
    if items.ndim != 2 or profiles.ndim != 2 or items.shape != profiles.shape:
        raise ValueError("item embeddings and user profiles must have equal 2D shapes")
    rows = torch.arange(items.shape[0])
    return fit_ranker_pair_encoder(
        model,
        items,
        profiles,
        rows,
        rows,
        fit_targets,
        config,
    )
