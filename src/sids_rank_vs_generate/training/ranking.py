from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from ..evaluation.ranking_metrics import ranking_metrics
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
) -> dict[str, Any]:
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
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
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

            batch_size = batch["target"].size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation = evaluate_ranking_model(
            model,
            validation_loader,
            device,
            use_bf16=config.use_bf16,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "validation": validation,
            }
        )
        if validation["roc_auc"] > best_auc:
            best_auc = validation["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("ranking training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, artifact_path)
    return {
        "device": str(device),
        "history": history,
        "test": evaluate_ranking_model(
            model,
            test_loader,
            device,
            use_bf16=config.use_bf16,
        ),
        "checkpoint": artifact_path,
    }
