""" Модуль обучает семантик трансформер, проверяет его на валидационной выборке, 
сохраняет лучшую модель по реколу и останавливает обучение, если качество долго не улучшается. 
В конце берём лучшую модель по реколу
Сейчас тут два режима валидации с коллизиями и без"""

from pathlib import Path

import torch

from sids_rank_vs_generate.evaluation.semantic_validation_with_collision import (
    validate_semantic_model_with_collisions,
)
from sids_rank_vs_generate.evaluation.semantic_validation_without_collision import (
    validate_semantic_model_without_collisions,
)
from sids_rank_vs_generate.training.utils import autocast, move_to_device
from sids_rank_vs_generate.tracking import ExperimentTracker, NullTracker


def train_semantic(
    model,
    train_loader,
    val_loader,
    cfg,
    sid_to_items=None,
    sid_catalog=None,
    sid_logq_table=None,
    tracker: ExperimentTracker | None = None,
    validation_callback=None,
):
    tracker = tracker or NullTracker()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    checkpoint_path = Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if validation_callback is not None:
        validate_fn = validation_callback
        validation_kwargs = {}
    elif cfg.resolve_collisions:
        validate_fn = validate_semantic_model_without_collisions
        validation_kwargs = {
            "sid_catalog": sid_catalog,
            "sid_logq_table": sid_logq_table,
            "sid_to_items": sid_to_items,
        }
    else:
        if sid_to_items is None:
            raise ValueError(
                "Для использования режима с коллизиями нужно передать маппер их сидов в айтемы: sid_to_items"
            )

        validate_fn = validate_semantic_model_with_collisions
        validation_kwargs = {
            "sid_to_items": sid_to_items,
            "sid_catalog": sid_catalog,
            "sid_logq_table": sid_logq_table,
        }

    metric_name = getattr(cfg, "validation_metric", None) or (
        f"recall@{cfg.recall_k}"
    )
    best_metric = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sid_loss_sum = 0.0
        sid_sample_count = 0
        ranking_loss_sum = 0.0
        ranking_sample_count = 0
        sample_count = 0
        diagnostic_sums: dict[str, float] = {}
        diagnostic_batches = 0

        for batch in train_loader:
            batch = move_to_device(batch, cfg.device)
            model_kwargs = {}
            if "user_features" in batch:
                model_kwargs["user_features"] = batch["user_features"]

            with autocast(batch["history"].device, cfg.use_bf16):
                output = model(
                    batch["history"],
                    batch["positions"],
                    batch["mask"],
                    batch["sid"],
                    target=batch["target"],
                    log_q=batch.get("log_q"),
                    sid_log_q=batch.get("sid_log_q"),
                    **model_kwargs,
                )

            loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_size = batch["sid"].size(0)
            loss_sum += loss.item() * batch_size
            sid_loss = output.get("sid_loss")
            if sid_loss is not None:
                sid_loss_sum += sid_loss.item() * batch_size
                sid_sample_count += batch_size
            ranking_loss = output.get("ranking_loss")
            if ranking_loss is not None:
                ranking_loss_sum += ranking_loss.item() * batch_size
                ranking_sample_count += batch_size
            sample_count += batch_size
            diagnostics = output.get("ranking_diagnostics", {})
            if diagnostics:
                diagnostic_batches += 1
                for name, value in diagnostics.items():
                    diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(
                        value.detach().float().item()
                    )

        train_loss = loss_sum / max(sample_count, 1)

        metrics = validate_fn(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            **validation_kwargs,
        )

        try:
            current_metric = metrics[metric_name]
        except KeyError as error:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"validation metric {metric_name!r} is unavailable; "
                f"received: {available}"
            ) from error
        training_metrics = {"train.loss": train_loss}
        if sid_sample_count:
            training_metrics["train.sid_loss"] = sid_loss_sum / sid_sample_count
        if ranking_sample_count:
            training_metrics["train.ranking_loss"] = (
                ranking_loss_sum / ranking_sample_count
            )
        train_diagnostics = {
            f"train.logq.{name}": value / diagnostic_batches
            for name, value in diagnostic_sums.items()
            if diagnostic_batches
        }
        tracker.log_metrics(
            {
                **training_metrics,
                **train_diagnostics,
                **{
                    f"validation.{name}": value
                    for name, value in metrics.items()
                },
                f"validation.best_{metric_name}": max(best_metric, current_metric),
            },
            step=epoch,
        )

        print(
            f"epoch={epoch} | loss={train_loss:.6f} | "
            f"{metric_name}={current_metric:.6f} | best={best_metric:.6f}"
        )

        if current_metric > best_metric + cfg.early_stopping_min_delta:
            best_metric = current_metric
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": cfg,
                },
                checkpoint_path,
            )

            print("Saved new best model")
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
