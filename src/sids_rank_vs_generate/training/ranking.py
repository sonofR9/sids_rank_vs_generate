from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from ..evaluation.ranking_metrics import ranking_metrics
from ..tracking import ExperimentTracker, NullTracker
from .utils import autocast, move_to_device, resolve_device


@torch.inference_mode()
def evaluate_ranking_model(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    targets_all = []
    users_all = []
    timestamps_all = []
    for batch in loader:
        batch = move_to_device(batch, device)
        with autocast(device, use_bf16):
            logits = model(batch)
        logits_all.append(logits.float().cpu())
        targets_all.append(batch["target"].cpu())
        users_all.append(batch["user_idx"].cpu())
        timestamps_all.append(batch["candidate_timestamp"].cpu())
    return ranking_metrics(
        torch.cat(targets_all),
        torch.sigmoid(torch.cat(logits_all)),
        torch.cat(users_all),
        torch.cat(timestamps_all),
    )


def train_ranking_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    test_loader,
    config,
    artifact_path: Path,
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    tracker = tracker or NullTracker()
    device = resolve_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_auc = float("-inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(
            train_loader,
            desc=f"ranking epoch {epoch}/{config.epochs}",
            disable=not config.show_progress,
        )
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, config.use_bf16):
                logits = model(batch)
                loss = criterion(logits, batch["target"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            batch_size = batch["target"].numel()
            loss_sum += loss.item() * batch_size
            example_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = loss_sum / max(example_count, 1)
        validation = evaluate_ranking_model(
            model,
            validation_loader,
            device,
            use_bf16=config.use_bf16,
        )
        tracker.log_metrics(
            {
                "train.loss": train_loss,
                **{
                    f"validation.{name}": value
                    for name, value in validation.items()
                },
                "validation.best_roc_auc": max(best_auc, validation["roc_auc"]),
            },
            step=epoch,
        )
        if validation["roc_auc"] > best_auc:
            best_auc = validation["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("ranking training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, artifact_path)
    test_metrics = evaluate_ranking_model(
        model,
        test_loader,
        device,
        use_bf16=config.use_bf16,
    )
    tracker.log_metrics(
        {f"test.{name}": value for name, value in test_metrics.items()}
    )
    return {
        "device": str(device),
        "test": test_metrics,
        "checkpoint": artifact_path,
    }
