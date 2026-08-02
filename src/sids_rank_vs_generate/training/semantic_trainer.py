
""" Модуль обучает семантик трансформер, проверяет его на валидационной выборке, 
сохраняет лучшую модель по реколу и останавливает обучение, если качество долго не улучшается. 
В конце берём лучшую модель по реколу
Сейчас тут два режима валидации с коллизиями и без"""
from pathlib import Path

import torch

from sids_rank_vs_generate.evaluation.semantic_validation_with_collision import validate_semantic_model_with_collisions
from sids_rank_vs_generate.evaluation.semantic_validation_without_collision import to_device, validate_semantic_model_without_collisions


def train_semantic(model, train_loader, val_loader, cfg, sid_to_items=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    checkpoint_path = Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.resolve_collisions:
        validate_fn = validate_semantic_model_without_collisions
        validation_kwargs = {}
    else:
        if sid_to_items is None:
            raise ValueError(
                "Для использования режима с коллизиями нужно передать маппер их сидов в айтемы: sid_to_items")

        validate_fn = validate_semantic_model_with_collisions
        validation_kwargs = {"sid_to_items": sid_to_items}

    metric_name = f"recall@{cfg.recall_k}"
    best_metric = float("-inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for batch in train_loader:
            batch = to_device(batch, cfg.device)

            output = model(
                batch["history"],
                batch["positions"],
                batch["mask"],
                batch["sid"],
            )

            loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_size = batch["sid"].size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

        train_loss = loss_sum / max(sample_count, 1)

        metrics = validate_fn(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            **validation_kwargs,
        )

        current_metric = metrics[metric_name]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        })

        print(
            f"epoch={epoch} | loss={train_loss:.6f} | "
            f"{metric_name}={current_metric:.6f} | best={best_metric:.6f}"
        )

        if current_metric > best_metric + cfg.early_stopping_min_delta:
            best_metric = current_metric
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": cfg,
            }, checkpoint_path)
            continue

        epochs_without_improvement += 1

        print(
            f"No improvement: {epochs_without_improvement}/"
            f"{cfg.early_stopping_patience}"
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print("Early stopping")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return history