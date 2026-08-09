from typing import Any

import numpy as np
import torch

from ..training.utils import autocast, move_to_device
from .beam_search import semantic_beam_search


# просто беру иду по всем предсказанным сидам, если у кого-то коллизия, то их все выписываю все, беру топ-к айтемомов
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
):
    model.eval()

    total = 0
    recall_sum = 0.0

    token_correct = 0
    token_total = 0
    top1_exact = 0
    all_predictions = []
    for batch in val_loader:
        batch = move_to_device(batch, cfg.device)

        with autocast(batch["history"].device, cfg.use_bf16):
            out = model(
                batch["history"],
                batch["positions"],
                batch["mask"],
                batch["sid"],
                target=batch["target"],
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
        )

        for user_beams, target_item in zip(beams, batch["target"]):
            predicted_items = beams_to_items(
                user_beams=user_beams,
                sid_to_items=sid_to_items,
                k=cfg.recall_k,
            )
            all_predictions.extend(predicted_items)
            recall_sum += float(int(target_item.item()) in set(predicted_items))
            total += 1
    _, counts = np.unique(all_predictions, return_counts=True)
    idx = np.argsort(counts)[::-1]
    sorted_counts = counts[idx]
    cumsum = np.cumsum(sorted_counts)
    prediction_total = np.sum(counts)
    coverage_ratio = cumsum / max(prediction_total, 1)
    threshold = 0.95
    idx_95 = (
        np.searchsorted(coverage_ratio, threshold) + 1
        if prediction_total
        else 0
    )
    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recall@{cfg.recall_k}": recall_sum / max(total, 1),
        "beam_size": cfg.num_beams,
        "Effective vocab size": idx_95,
    }
