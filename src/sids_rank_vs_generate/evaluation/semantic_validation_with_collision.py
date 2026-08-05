from typing import Any


import torch

from .beam_search import semantic_beam_search


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }

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
):
    model.eval()

    total = 0
    recall_sum = 0.0

    token_correct = 0
    token_total = 0
    top1_exact = 0

    for batch in val_loader:
        batch = to_device(batch, cfg.device)

        out = model(
            batch["history"],
            batch["positions"],
            batch["mask"],
            batch["sid"],
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
        )

        for user_beams, target_items in zip(beams, batch["target"]):
            predicted_items = beams_to_items(
                user_beams=user_beams,
                sid_to_items=sid_to_items,
                k=cfg.recall_k,
            )

            target_items = set(target_items[target_items >= 0].tolist())
            hits = len(set(predicted_items) & target_items)
            recall_sum += hits / len(target_items)
            total += 1

    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recall@{cfg.recall_k}": recall_sum / max(total, 1),
        "beam_size": cfg.num_beams,
    }