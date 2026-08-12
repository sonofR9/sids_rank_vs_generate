from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from tqdm.auto import tqdm

from ..sids.user_sid_mapper import IUserSidMapper
from ..training.utils import autocast, move_to_device
from .beam_search import SemanticIDCatalog, semantic_beam_search


def _append_unique_mapped_items(
    ranked_items: list[int],
    seen_items: set[int],
    mapped_paths_by_beam: Sequence[Sequence[tuple[int, ...]]],
    sid_to_items: dict[tuple[int, ...], list[int]],
    *,
    limit: int | None,
) -> set[tuple[int, ...]]:
    global_candidates: set[tuple[int, ...]] = set()
    for global_sids in mapped_paths_by_beam:
        if limit is not None and len(ranked_items) >= limit:
            break
        for global_sid in global_sids:
            path = tuple(int(code) for code in global_sid)
            global_candidates.add(path)
            for item in sid_to_items.get(path, []):
                if item not in seen_items:
                    seen_items.add(item)
                    ranked_items.append(item)
                if limit is not None and len(ranked_items) >= limit:
                    break
            if limit is not None and len(ranked_items) >= limit:
                break
    return global_candidates


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
            _append_unique_mapped_items(
                ranked_items[user_index],
                seen_items[user_index],
                user_predictions,
                sid_to_items,
                limit=recall_k,
            )
            if len(ranked_items[user_index]) < recall_k:
                remaining_users.append(user_index)
        active_users = remaining_users

    return ranked_items


@torch.inference_mode()
def validate_personalized_generation(
    model,
    loader,
    cfg,
    mapper: IUserSidMapper | None = None,
    sid_to_items: dict[tuple[int, ...], list[int]] | None = None,
    sid_catalog: SemanticIDCatalog | None = None,
    comparison_mappers: Mapping[str, IUserSidMapper] | None = None,
) -> dict[str, float | int]:
    if mapper is None and comparison_mappers:
        raise ValueError("comparison mappers require the primary mapper")
    if (mapper is None) != (sid_to_items is None):
        raise ValueError("mapper and sid_to_items must be provided together")
    model.eval()
    total = 0
    hits = 0
    reciprocal_rank = 0.0
    token_correct = 0
    token_total = 0
    top1_exact = 0
    sid_beam_hits = 0
    sid_beam_reciprocal_rank = 0.0
    sid_beam_size_min = None
    sid_beam_size_max = 0
    sid_cutoffs = tuple(
        sorted({1, 3, 10, 50, 100, int(cfg.num_beams)})
    )
    sid_hits_by_cutoff = {cutoff: 0 for cutoff in sid_cutoffs}
    sid_prefix_hits = [0] * int(cfg.num_levels)
    sid_top1_token_hits = [0] * int(cfg.num_levels)
    recommended_items: set[int] = set()
    comparison = {
        name: {"hits": 0, "reciprocal_rank": 0.0, "items": set()}
        for name in (comparison_mappers or {})
    }

    for batch in tqdm(
        loader,
        desc="personalized evaluation",
        unit="batch",
        mininterval=5.0,
    ):
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
                **model_kwargs,
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
            **model_kwargs,
        )
        sid_beam_size_min = (
            beams.size(1)
            if sid_beam_size_min is None
            else min(sid_beam_size_min, beams.size(1))
        )
        sid_beam_size_max = max(sid_beam_size_max, beams.size(1))
        beam_matches = (beams == batch["sid"].unsqueeze(1)).all(dim=2)
        beam_hit = beam_matches.any(dim=1)
        beam_rank = beam_matches.to(torch.int64).argmax(dim=1) + 1
        sid_beam_hits += beam_hit.sum().item()
        sid_beam_reciprocal_rank += (
            beam_hit.float() / beam_rank.float()
        ).sum().item()
        for cutoff in sid_cutoffs:
            sid_hits_by_cutoff[cutoff] += beam_matches[
                :, : min(cutoff, beams.size(1))
            ].any(dim=1).sum().item()
        for level in range(int(cfg.num_levels)):
            prefix_length = level + 1
            prefix_matches = (
                beams[:, :, :prefix_length]
                == batch["sid"].unsqueeze(1)[:, :, :prefix_length]
            ).all(dim=2)
            sid_prefix_hits[level] += prefix_matches.any(dim=1).sum().item()
            sid_top1_token_hits[level] += (
                beams[:, 0, level] == batch["sid"][:, level]
            ).sum().item()
        if mapper is not None and sid_to_items is not None:
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
        total += batch["sid"].size(0)

        for name, comparison_mapper in (comparison_mappers or {}).items():
            comparison_ranked = _rank_personalized_items(
                uids=batch["uid"],
                beams=beams,
                mapper=comparison_mapper,
                sid_to_items=sid_to_items,
                recall_k=cfg.recall_k,
                mapper_top_k=cfg.mapper_top_k,
                mapper_beam_chunk_size=cfg.mapper_beam_chunk_size,
            )
            values = comparison[name]
            for ranked_items, target in zip(
                comparison_ranked,
                batch["target"],
                strict=True,
            ):
                values["items"].update(ranked_items)
                target_item = int(target.item())
                if target_item in ranked_items:
                    rank = ranked_items.index(target_item) + 1
                    values["hits"] += 1
                    values["reciprocal_rank"] += 1.0 / rank

    beam_size = int(cfg.num_beams)
    result = {
        "top1_sid_token_acc": token_correct / max(token_total, 1),
        "top1_sid_exact": top1_exact / max(total, 1),
        f"sid_beam_recall@{beam_size}": sid_beam_hits / max(total, 1),
        f"sid_beam_mrr@{beam_size}": sid_beam_reciprocal_rank / max(total, 1),
        "sid_beam_effective_size_min": sid_beam_size_min or 0,
        "sid_beam_effective_size_max": sid_beam_size_max,
        "beam_size": cfg.num_beams,
    }
    for cutoff, cutoff_hits in sid_hits_by_cutoff.items():
        result[f"sid_beam_recall@{cutoff}"] = cutoff_hits / max(total, 1)
    for level, (prefix_hits, token_hits) in enumerate(
        zip(sid_prefix_hits, sid_top1_token_hits, strict=True),
        start=1,
    ):
        result[f"sid_prefix_recall@{beam_size}_level_{level}"] = (
            prefix_hits / max(total, 1)
        )
        result[f"sid_top1_token_accuracy_level_{level}"] = (
            token_hits / max(total, 1)
        )
    if mapper is not None and sid_to_items is not None:
        catalog = {item for items in sid_to_items.values() for item in items}
        result[f"recall@{cfg.recall_k}"] = hits / max(total, 1)
        result[f"mrr@{cfg.recall_k}"] = reciprocal_rank / max(total, 1)
        result[f"catalog_coverage@{cfg.recall_k}"] = (
            len(recommended_items) / max(len(catalog), 1)
        )
    else:
        catalog = set()
    for name, values in comparison.items():
        result[f"{name}_recall@{cfg.recall_k}"] = values["hits"] / max(total, 1)
        result[f"{name}_mrr@{cfg.recall_k}"] = values["reciprocal_rank"] / max(
            total,
            1,
        )
        result[f"{name}_catalog_coverage@{cfg.recall_k}"] = len(
            values["items"]
        ) / max(len(catalog), 1)
    return result
