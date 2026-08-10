from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from ..evaluation.ranking_metrics import ranking_metrics
from ..tracking import ExperimentTracker, NullTracker
from .utils import math_sdp_only, move_to_device, resolve_device


def prefetch_batches(loader):
    """Prepare the next columnar CPU batch while the GPU handles this one."""

    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ranking-batch") as pool:
        future = pool.submit(next, iterator, None)
        while True:
            batch = future.result()
            if batch is None:
                return
            future = pool.submit(next, iterator, None)
            yield batch


@torch.inference_mode()
def evaluate_ranking_model(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    logits_all = []
    targets_all = []
    users_all = []
    timestamps_all = []
    for batch in prefetch_batches(loader):
        batch = move_to_device(batch, device)
        with math_sdp_only(device):
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
    torch.set_float32_matmul_precision("high")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_validation = None
    best_pairwise_accuracy = float("-inf")
    best_auc = float("-inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(
            prefetch_batches(train_loader),
            total=len(train_loader),
            desc=f"ranking epoch {epoch}/{config.epochs}",
            disable=not config.show_progress,
        )
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with math_sdp_only(device):
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
        )
        tracker.log_metrics(
            {
                "train.loss": train_loss,
                **{
                    f"validation.{name}": value
                    for name, value in validation.items()
                },
                "validation.best_pairwise_accuracy": max(
                    best_pairwise_accuracy,
                    validation["pairwise_accuracy"],
                ),
                "validation.best_roc_auc": max(best_auc, validation["roc_auc"]),
            },
            step=epoch,
        )
        best_auc = max(best_auc, validation["roc_auc"])
        if validation["pairwise_accuracy"] > best_pairwise_accuracy:
            best_pairwise_accuracy = validation["pairwise_accuracy"]
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("ranking training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, artifact_path)
    test_metrics = evaluate_ranking_model(
        model,
        test_loader,
        device,
    )
    tracker.log_metrics(
        {f"test.{name}": value for name, value in test_metrics.items()}
    )
    return {
        "device": str(device),
        "primary_metric": "pairwise_accuracy",
        "best_validation": best_validation,
        "best_validation_pairwise_accuracy": best_pairwise_accuracy,
        "test": test_metrics,
        "checkpoint": artifact_path,
    }
