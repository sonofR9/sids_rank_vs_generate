from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from ..sids.user_sid_mapper import IUserSidMapper
from ..training.utils import autocast, move_to_device
from .beam_search import SemanticIDCatalog, semantic_beam_search


def _rank_personalized_items(
    *,
    uids: Sequence[object],
    beams: torch.Tensor,
    mapper: IUserSidMapper,
    sid_to_items: dict[tuple[int, ...], list[int]],
    recall_k: int,
    mapper_top_k: int,
    mapper_beam_chunk_size: int,
) -> list[list[int]]:
    if mapper_beam_chunk_size <= 0:
        raise ValueError("mapper_beam_chunk_size must be positive")
    if beams.ndim != 3 or beams.size(0) != len(uids):
        raise ValueError("personalized beams have an unexpected shape")

    beam_paths = beams.detach().cpu().tolist()
    ranked_items = [[] for _ in uids]
    seen_items = [set() for _ in uids]
    active_users = list(range(len(uids)))

    for beam_offset in range(0, beams.size(1), mapper_beam_chunk_size):
        if not active_users:
            break
        beam_end = min(beam_offset + mapper_beam_chunk_size, beams.size(1))
        requests = [
            (uids[user_index], beam_paths[user_index][beam_index])
            for user_index in active_users
            for beam_index in range(beam_offset, beam_end)
        ]
        mapped_paths = mapper.predict_topk_batch(requests, mapper_top_k)
        if len(mapped_paths) != len(requests):
            raise RuntimeError("mapper returned an unexpected number of predictions")

        mapper_offset = 0
        remaining_users = []
        for user_index in active_users:
            user_predictions = mapped_paths[
                mapper_offset : mapper_offset + beam_end - beam_offset
            ]
            mapper_offset += beam_end - beam_offset
            for global_sids in user_predictions:
                if len(ranked_items[user_index]) >= recall_k:
                    break
                for global_sid in global_sids:
                    for item in sid_to_items.get(global_sid, []):
                        if item not in seen_items[user_index]:
                            seen_items[user_index].add(item)
                            ranked_items[user_index].append(item)
                        if len(ranked_items[user_index]) >= recall_k:
                            break
                    if len(ranked_items[user_index]) >= recall_k:
                        break
            if len(ranked_items[user_index]) < recall_k:
                remaining_users.append(user_index)
        active_users = remaining_users

    return ranked_items


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

    for batch in tqdm(
        loader,
        desc="personalized evaluation",
        unit="batch",
        mininterval=5.0,
    ):
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
        ranked_items_by_user = _rank_personalized_items(
            uids=batch["uid"],
            beams=beams,
            mapper=mapper,
            sid_to_items=sid_to_items,
            recall_k=cfg.recall_k,
            mapper_top_k=cfg.mapper_top_k,
            mapper_beam_chunk_size=cfg.mapper_beam_chunk_size,
        )
        for ranked_items, target in zip(
            ranked_items_by_user,
            batch["target"],
            strict=True,
        ):
            recommended_items.update(ranked_items)
            seen = set(ranked_items)
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
