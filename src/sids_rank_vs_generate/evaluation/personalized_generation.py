from __future__ import annotations

import torch

from ..sids.user_sid_mapper import IUserSidMapper
from ..training.utils import autocast, move_to_device
from .beam_search import SemanticIDCatalog, semantic_beam_search


@torch.inference_mode()
def validate_personalized_generation(
    model,
    loader,
    cfg,
    mapper: IUserSidMapper,
    sid_to_items: dict[tuple[int, ...], list[int]],
    sid_catalog: SemanticIDCatalog | None = None,
) -> dict[str, float | int]:
    model.eval()
    total = 0
    hits = 0
    reciprocal_rank = 0.0
    token_correct = 0
    token_total = 0
    top1_exact = 0
    recommended_items: set[int] = set()

    for batch in loader:
        batch = move_to_device(batch, cfg.device)
        with autocast(batch["history"].device, cfg.use_bf16):
            output = model(
                batch["history"],
                batch["positions"],
                batch["mask"],
                batch["sid"],
                target=batch["target"],
            )
        predictions = output["preds"]
        token_correct += (predictions == batch["sid"]).sum().item()
        token_total += batch["sid"].numel()
        top1_exact += (predictions == batch["sid"]).all(dim=1).sum().item()

        beams, _ = semantic_beam_search(
            model=model,
            history=batch["history"],
            positions=batch["positions"],
            mask=batch["mask"],
            beam_size=cfg.num_beams,
            use_bf16=cfg.use_bf16,
            sid_catalog=sid_catalog,
        )
        mapper_requests = [
            (uid, beam.tolist())
            for uid, user_beams in zip(batch["uid"], beams, strict=True)
            for beam in user_beams
        ]
        mapped_paths = mapper.predict_topk_batch(
            mapper_requests,
            cfg.mapper_top_k,
        )
        mapper_offset = 0
        for uid, user_beams, target in zip(
            batch["uid"],
            beams,
            batch["target"],
            strict=True,
        ):
            ranked_items: list[int] = []
            seen: set[int] = set()
            for beam in user_beams:
                global_sids = mapped_paths[mapper_offset]
                mapper_offset += 1
                for global_sid in global_sids:
                    for item in sid_to_items.get(global_sid, []):
                        if item not in seen:
                            seen.add(item)
                            ranked_items.append(item)
                        if len(ranked_items) >= cfg.recall_k:
                            break
                    if len(ranked_items) >= cfg.recall_k:
                        break
                if len(ranked_items) >= cfg.recall_k:
                    break

            recommended_items.update(ranked_items)
            target_item = int(target.item())
            if target_item in seen:
                rank = ranked_items.index(target_item) + 1
                hits += 1
                reciprocal_rank += 1.0 / rank
            total += 1

    catalog = {item for items in sid_to_items.values() for item in items}
    return {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"recall@{cfg.recall_k}": hits / max(total, 1),
        f"mrr@{cfg.recall_k}": reciprocal_rank / max(total, 1),
        f"catalog_coverage@{cfg.recall_k}": (
            len(recommended_items) / max(len(catalog), 1)
        ),
        "beam_size": cfg.num_beams,
    }
