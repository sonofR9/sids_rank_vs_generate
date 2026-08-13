from typing import Any

import numpy as np
import torch

from ..training.utils import autocast, move_to_device
from .beam_search import semantic_beam_search


# Разворачиваем SID с коллизиями в top-k айтемов
def beams_to_items(user_beams, sid_to_items, k):
    predicted_items = []

    for beam in user_beams:
        sid = tuple(beam.tolist())
        predicted_items.extend(sid_to_items.get(sid, []))

        if len(predicted_items) >= k:
            return predicted_items[:k]

    return predicted_items


@torch.inference_mode()
def validate_semantic_model_with_collisions(
    model,
    val_loader,
    cfg,
    sid_to_items,
    sid_catalog=None,
    sid_logq_table=None,
):
    model.eval()

    total = 0
    recall_sum = 0.0

    token_correct = 0
    token_total = 0
    top1_exact = 0
    all_predictions = []
    reciprocal_rank = 0.0
    recommended_items = set()
    for batch in val_loader:
        batch = move_to_device(batch, cfg.device)
        model_kwargs = {}
        if "user_features" in batch:
            model_kwargs["user_features"] = batch["user_features"]
        if "sid_log_q" in batch:
            model_kwargs["sid_log_q"] = batch["sid_log_q"]

        with autocast(batch["history"].device, cfg.use_bf16):
            out = model(
                batch["history"],
                batch["positions"],
                batch["mask"],
                batch["sid"],
                **model_kwargs,
            )

        preds = out["preds"]

        token_correct += (preds == batch["sid"]).sum().item()
        token_total += batch["sid"].numel()
        top1_exact += (preds == batch["sid"]).all(dim=1).sum().item()

        beams, _ = semantic_beam_search(
            model=model,
            history=batch["history"],
            positions=batch["positions"],
            mask=batch["mask"],
            beam_size=cfg.num_beams,
            use_bf16=cfg.use_bf16,
            sid_catalog=sid_catalog,
            sid_logq_table=sid_logq_table,
            user_features=model_kwargs.get("user_features"),
        )

        for user_beams, target_item in zip(beams, batch["target"]):
            predicted_items = beams_to_items(
                user_beams=user_beams,
                sid_to_items=sid_to_items,
                k=cfg.recall_k,
            )
            all_predictions.extend(predicted_items)
            recall_sum += float(int(target_item.item()) in set(predicted_items))
            recommended_items.update(predicted_items)
            if int(target_item.item()) in predicted_items:
                reciprocal_rank += 1.0 / (
                    predicted_items.index(int(target_item.item())) + 1
                )
            total += 1
    _, counts = np.unique(all_predictions, return_counts=True)
    idx = np.argsort(counts)[::-1]
    sorted_counts = counts[idx]
    cumsum = np.cumsum(sorted_counts)
    prediction_total = np.sum(counts)
    coverage_ratio = cumsum / max(prediction_total, 1)
    threshold = 0.95
    idx_95 = np.searchsorted(coverage_ratio, threshold) + 1 if prediction_total else 0

    catalog = {item for items in sid_to_items.values() for item in items}
    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recall@{cfg.recall_k}": recall_sum / max(total, 1),
        f"mrr@{cfg.recall_k}": reciprocal_rank / max(total, 1),
        f"catalog_coverage@{cfg.recall_k}": (
            len(recommended_items) / max(len(catalog), 1)
        ),
        "beam_size": cfg.num_beams,
        "Effective vocab size": idx_95,
    }
