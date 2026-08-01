from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from tqdm.auto import tqdm

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class ExplicitFeedbackTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        use_amp=True,
        grad_clip_norm=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm

        self.history = []

        self.best_state_dict = None
        self.best_metric_value = None

    def move_batch_to_device(self, batch):
        return {
            key: value.to(
                self.device,
                non_blocking=True,
            )
            for key, value in batch.items()
        }

    def forward_step(self, batch):
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key != "target"
        }

        logits = self.model(**model_inputs)

        loss = self.criterion(logits, batch["target"])

        return logits, loss

    def update_step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
            )

        self.optimizer.step()

    def train_epoch(self, loader, epoch):
        self.model.train()

        total_loss = 0.0
        total_examples = 0

        pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

        for batch in pbar:
            batch = self.move_batch_to_device(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    self.use_amp
                    and self.device.type == "cuda")):
                _, loss = self.forward_step(batch)

            self.update_step(loss)

            batch_size = batch["target"].size(0)

            total_loss += (loss.item() * batch_size)

            total_examples += batch_size

            pbar.set_postfix(bce=total_loss / total_examples)

        return total_loss / total_examples

    @torch.no_grad()
    def eval_epoch(
        self,
        loader,
        split_name="val",
    ):
        self.model.eval()

        total_loss = 0.0
        total_examples = 0

        all_targets = []
        all_prob_like = []

        pbar = tqdm(
            loader,
            desc=split_name,
            leave=False
        )

        for batch in pbar:
            batch = self.move_batch_to_device(
                batch
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    self.use_amp
                    and self.device.type == "cuda")
            ):
                logits, loss = self.forward_step(
                    batch
                )

            prob_like = torch.sigmoid(logits)

            batch_size = batch["target"].size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            all_targets.append(
                batch["target"]
                .detach()
                .float()
                .cpu()
            )

            all_prob_like.append(
                prob_like
                .detach()
                .float()
                .cpu()
            )

        y_like = torch.cat(
            all_targets
        ).numpy()

        p_like = torch.cat(
            all_prob_like
        ).numpy()

        p_like = np.clip(
            p_like,
            1e-7,
            1 - 1e-7,
        )

        y_dislike = 1 - y_like
        p_dislike = 1 - p_like

        pred_like = (
            p_like >= 0.5
        ).astype(np.int64)

        metrics = {
            "bce": (
                total_loss / total_examples
            ),

            "roc_auc": roc_auc_score(
                y_like,
                p_like,
            ),

            "pr_auc_like": (
                average_precision_score(
                    y_like,
                    p_like,
                )
            ),

            "pr_auc_dislike": (
                average_precision_score(
                    y_dislike,
                    p_dislike,
                )
            ),

            "logloss": log_loss(
                y_like,
                p_like,
                labels=[0, 1],
            ),

            "balanced_accuracy": (
                balanced_accuracy_score(
                    y_like,
                    pred_like,
                )
            ),

            "macro_f1": f1_score(
                y_like,
                pred_like,
                average="macro",
                zero_division=0,
            ),
        }

        return metrics

    def is_better(
        self,
        current,
        best,
        mode,
    ):
        if best is None:
            return True

        if mode == "max":
            return current > best

        if mode == "min":
            return current < best

        raise ValueError(
            "mode must be 'max' or 'min'"
        )

    def train(
        self,
        train_loader,
        val_loader,
        epochs=5,
        monitor_metric="roc_auc",
        monitor_mode="max",
    ):
        for epoch in range(1, epochs + 1):
            train_bce = self.train_epoch(train_loader, epoch)

            val_metrics = self.eval_epoch(val_loader, split_name="val",)

            row = {
                "epoch": epoch,
                "train_bce": train_bce,
                **{
                    f"val_{key}": value
                    for key, value
                    in val_metrics.items()
                },
            }

            self.history.append(row)

            current = val_metrics[monitor_metric]

            if self.is_better(
                current,
                self.best_metric_value,
                monitor_mode
            ):
                self.best_metric_value = current

                self.best_state_dict = (
                    copy.deepcopy(
                        self.model.state_dict()
                    )
                )

            print(
                f"epoch {epoch:02d} | "
                f"train_bce={train_bce:.4f} | "
                f"val_bce={val_metrics['bce']:.4f} | "
                f"roc_auc={val_metrics['roc_auc']:.4f} | "
                f"pr_like={val_metrics['pr_auc_like']:.4f} | "
                f"pr_dislike={val_metrics['pr_auc_dislike']:.4f}"
            )

        return pd.DataFrame(self.history)

    def load_best_model(self):
        if self.best_state_dict is None:
            raise RuntimeError(
                "No best model has been saved."
            )

        self.model.load_state_dict(
            self.best_state_dict
        )


def make_optimizer(
    model,
    lr=3e-4,
    weight_decay=1e-4,
):
    device = next(model.parameters()).device

    if device.type == "cuda":
        try:
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
            )
        except (TypeError, RuntimeError):
            pass

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
