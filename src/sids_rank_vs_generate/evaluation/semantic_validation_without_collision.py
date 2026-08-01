import torch

from .beam_search import semantic_beam_search


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.inference_mode()
def validate_semantic_model_without_collisions(
    model,
    val_loader,
    cfg,
):
    model.eval()

    total = 0
    exact_hits = 0
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

        beams = beams[:, :cfg.recall_k]

        hit = (beams == batch["sid"][:, None, :]).all(dim=-1).any(dim=1)

        total += batch["sid"].size(0)
        exact_hits += hit.sum().item()

    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recallt@{cfg.recall_k}": exact_hits / max(total, 1),
        "beam_size": cfg.num_beams,
    }