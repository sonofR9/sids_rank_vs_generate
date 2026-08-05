import torch

from ..training.utils import autocast, move_to_device
from .beam_search import semantic_beam_search
from collections import Counter

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
    freq_counter = Counter()
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
        )

        beams = beams[:, : cfg.recall_k]
        for user_beams in beams:
            for sid_tokens in user_beams:
                freq_counter[tuple(sid_tokens.tolist())] += 1
        hit = (beams == batch["sid"][:, None, :]).all(dim=-1).any(dim=1)

        total += batch["sid"].size(0)
        exact_hits += hit.sum().item()
    sorted_freqs = sorted(freq_counter.values(), reverse=True)
    total_predictions = sum(sorted_freqs)
    cumsum = 0
    items_needed = 0
    for freq in sorted_freqs:
        cumsum += freq
        items_needed += 1
        if cumsum / total_predictions >= 0.95:
            break
    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recall@{cfg.recall_k}": exact_hits / max(total, 1),
        "beam_size": cfg.num_beams,
        "Effective vocab size": items_needed,
    }
