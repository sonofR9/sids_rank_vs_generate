from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from ..sids.user_sid_mapper import IUserSidMapper, SidPath
from ..training.utils import move_to_device
from .beam_search import SemanticIDCatalog, semantic_beam_search
from .personalized_generation import _append_unique_mapped_items


DEFAULT_BEAM_CUTOFFS = (1, 3, 10, 50, 100, 150)
DEFAULT_MAPPER_CUTOFFS = (1, 3, 5, 10, 20)
DEFAULT_RECOMMENDATION_CUTOFFS = (10, 50, 100)
_STRATA = ("all", "seen", "unseen")


def _positive_cutoffs(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive values")
    return result


@dataclass
class _StratifiedRate:
    hits: dict[str, int] = field(default_factory=lambda: {name: 0 for name in _STRATA})
    examples: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _STRATA}
    )

    def add(self, value: bool, *, seen: bool, include: bool = True) -> None:
        if not include:
            return
        names = ("all", "seen" if seen else "unseen")
        for name in names:
            self.examples[name] += 1
            self.hits[name] += int(value)

    def result(self) -> dict[str, dict[str, int | float | None]]:
        return {
            name: {
                "hits": self.hits[name],
                "examples": self.examples[name],
                "rate": (
                    self.hits[name] / self.examples[name]
                    if self.examples[name]
                    else None
                ),
            }
            for name in _STRATA
        }


@dataclass
class _StratifiedMean:
    total: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _STRATA}
    )
    examples: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _STRATA}
    )

    def add(self, value: int | float, *, seen: bool) -> None:
        names = ("all", "seen" if seen else "unseen")
        for name in names:
            self.total[name] += float(value)
            self.examples[name] += 1

    def result(self) -> dict[str, dict[str, int | float | None]]:
        return {
            name: {
                "examples": self.examples[name],
                "mean": (
                    self.total[name] / self.examples[name]
                    if self.examples[name]
                    else None
                ),
            }
            for name in _STRATA
        }


def _rate_difference(
    candidate: _StratifiedRate,
    final: _StratifiedRate,
) -> dict[str, dict[str, int | float | None]]:
    result = {}
    for name in _STRATA:
        if candidate.examples[name] != final.examples[name]:
            raise RuntimeError("ordering-loss rates have different denominators")
        candidate_hits = candidate.hits[name]
        final_hits = final.hits[name]
        lost_hits = candidate_hits - final_hits
        if lost_hits < 0:
            raise RuntimeError("final recall cannot exceed candidate coverage")
        examples = candidate.examples[name]
        result[name] = {
            "candidate_hits": candidate_hits,
            "final_hits": final_hits,
            "lost_hits": lost_hits,
            "examples": examples,
            "rate": lost_hits / examples if examples else None,
        }
    return result


def _ordered_unique_items(
    mapped_paths_by_beam: list[list[SidPath]],
    sid_to_items: dict[SidPath, list[int]],
    *,
    mapper_top_k: int,
) -> tuple[list[int], set[SidPath]]:
    ranked_items: list[int] = []
    seen_items: set[int] = set()
    global_candidates = _append_unique_mapped_items(
        ranked_items,
        seen_items,
        [mapped_paths[:mapper_top_k] for mapped_paths in mapped_paths_by_beam],
        sid_to_items,
        limit=None,
    )
    return ranked_items, global_candidates


def _split_mapper_results(
    values: list[list[SidPath]],
    *,
    batch_size: int,
    beam_count: int,
) -> list[list[list[SidPath]]]:
    expected = batch_size * beam_count
    if len(values) != expected:
        raise RuntimeError(
            f"mapper returned {len(values)} generated-beam results, expected {expected}"
        )
    return [
        values[index * beam_count : (index + 1) * beam_count]
        for index in range(batch_size)
    ]


def mapper_inference_mlflow_summary(result: dict[str, Any]) -> dict[str, Any]:
    recommendation_key = "100"
    grid = result["grid"]
    return {
        "E_B": {beam: values["all"]["rate"] for beam, values in result["E_B"].items()},
        "teacher_forced_mapper_top_k": {
            mapper_k: values["all"]["rate"]
            for mapper_k, values in result["teacher_forced_mapper_top_k"].items()
        },
        "grid": {
            beam: {
                mapper_k: {
                    "item_candidate_coverage": values[
                        "item_candidate_coverage_before_truncation"
                    ]["all"]["rate"],
                    "final_recall_at_100": values["final_recall"][recommendation_key][
                        "all"
                    ]["rate"],
                    "ordering_loss_at_100": values["ordering_loss"][recommendation_key][
                        "all"
                    ]["rate"],
                }
                for mapper_k, values in beam_values.items()
            }
            for beam, beam_values in grid.items()
        },
    }


@torch.inference_mode()
def diagnose_mapper_inference(
    model,
    loader,
    cfg,
    mapper: IUserSidMapper,
    sid_to_items: dict[SidPath, list[int]],
    *,
    sid_catalog: SemanticIDCatalog | None = None,
    beam_cutoffs: Iterable[int] = DEFAULT_BEAM_CUTOFFS,
    mapper_cutoffs: Iterable[int] = DEFAULT_MAPPER_CUTOFFS,
    recommendation_cutoffs: Iterable[int] = DEFAULT_RECOMMENDATION_CUTOFFS,
) -> dict[str, Any]:
    """Measure mapper gates and the real SID-to-item ordering without training."""

    beam_cutoffs = _positive_cutoffs(beam_cutoffs, "beam_cutoffs")
    mapper_cutoffs = _positive_cutoffs(mapper_cutoffs, "mapper_cutoffs")
    recommendation_cutoffs = _positive_cutoffs(
        recommendation_cutoffs,
        "recommendation_cutoffs",
    )
    max_beams = max(beam_cutoffs)

    example_counts = {name: 0 for name in _STRATA}
    event_rates = {cutoff: _StratifiedRate() for cutoff in beam_cutoffs}
    teacher_rates = {cutoff: _StratifiedRate() for cutoff in mapper_cutoffs}
    candidate_rates = {
        (beam, mapper_k): _StratifiedRate()
        for beam in beam_cutoffs
        for mapper_k in mapper_cutoffs
    }
    global_candidate_rates = {key: _StratifiedRate() for key in candidate_rates}
    conditional_candidate_rates = {key: _StratifiedRate() for key in candidate_rates}
    conditional_global_candidate_rates = {
        key: _StratifiedRate() for key in candidate_rates
    }
    conditional_teacher_rates = {key: _StratifiedRate() for key in candidate_rates}
    not_event_candidate_rates = {key: _StratifiedRate() for key in candidate_rates}
    not_event_global_candidate_rates = {
        key: _StratifiedRate() for key in candidate_rates
    }
    not_event_teacher_rates = {key: _StratifiedRate() for key in candidate_rates}
    joint_gate_rates = {key: _StratifiedRate() for key in candidate_rates}
    candidate_sizes = {key: _StratifiedMean() for key in candidate_rates}
    final_rates = {
        (beam, mapper_k, recommendation_k): _StratifiedRate()
        for beam in beam_cutoffs
        for mapper_k in mapper_cutoffs
        for recommendation_k in recommendation_cutoffs
    }
    conditional_final_rates = {key: _StratifiedRate() for key in final_rates}
    not_event_final_rates = {key: _StratifiedRate() for key in final_rates}
    gated_final_rates = {key: _StratifiedRate() for key in final_rates}
    effective_beam_min: int | None = None
    effective_beam_max = 0

    model.eval()
    for batch in tqdm(
        loader,
        desc="mapper inference diagnostic",
        unit="batch",
        mininterval=5.0,
    ):
        batch = move_to_device(batch, cfg.device)
        model_kwargs = {}
        if "user_features" in batch:
            model_kwargs["user_features"] = batch["user_features"]
        beams, _scores = semantic_beam_search(
            model=model,
            history=batch["history"],
            positions=batch["positions"],
            mask=batch["mask"],
            beam_size=max_beams,
            use_bf16=cfg.use_bf16,
            sid_catalog=sid_catalog,
            **model_kwargs,
        )
        batch_size, effective_beams, _levels = beams.shape
        effective_beam_min = (
            effective_beams
            if effective_beam_min is None
            else min(effective_beam_min, effective_beams)
        )
        effective_beam_max = max(effective_beam_max, effective_beams)

        uids = list(batch["uid"])
        true_user_sids = [
            tuple(int(code) for code in path)
            for path in batch["sid"].detach().cpu().tolist()
        ]
        true_global_sids = [
            tuple(int(code) for code in path)
            for path in batch["global_sid"].detach().cpu().tolist()
        ]
        target_items = [int(value) for value in batch["target"].detach().cpu().tolist()]
        beam_paths = [
            [tuple(int(code) for code in path) for path in sample]
            for sample in beams.detach().cpu().tolist()
        ]

        teacher_requests = list(zip(uids, true_user_sids, strict=True))
        generated_requests = [
            (uid, path)
            for uid, sample_beams in zip(uids, beam_paths, strict=True)
            for path in sample_beams
        ]
        teacher_predictions_by_k = {}
        generated_predictions_by_k = {}
        for mapper_k in mapper_cutoffs:
            teacher_predictions = mapper.predict_topk_batch(
                teacher_requests,
                mapper_k,
            )
            if len(teacher_predictions) != batch_size:
                raise RuntimeError(
                    "mapper returned an unexpected number of teacher-forced results"
                )
            teacher_predictions_by_k[mapper_k] = teacher_predictions
            generated_predictions_by_k[mapper_k] = _split_mapper_results(
                mapper.predict_topk_batch(generated_requests, mapper_k),
                batch_size=batch_size,
                beam_count=effective_beams,
            )

        for index in range(batch_size):
            uid = uids[index]
            true_user_sid = true_user_sids[index]
            true_global_sid = true_global_sids[index]
            target_item = target_items[index]
            sample_beams = beam_paths[index]
            seen = mapper.is_seen(uid, true_user_sid)
            example_counts["all"] += 1
            example_counts["seen" if seen else "unseen"] += 1

            event_by_beam = {}
            for beam in beam_cutoffs:
                event = true_user_sid in sample_beams[:beam]
                event_by_beam[beam] = event
                event_rates[beam].add(event, seen=seen)

            teacher_by_k = {}
            for mapper_k in mapper_cutoffs:
                sample_teacher = teacher_predictions_by_k[mapper_k][index]
                hit = true_global_sid in sample_teacher[:mapper_k]
                teacher_by_k[mapper_k] = hit
                teacher_rates[mapper_k].add(hit, seen=seen)

            for beam in beam_cutoffs:
                event = event_by_beam[beam]
                for mapper_k in mapper_cutoffs:
                    key = (beam, mapper_k)
                    sample_predictions = generated_predictions_by_k[mapper_k][index]
                    ranked_items, global_candidates = _ordered_unique_items(
                        sample_predictions[:beam],
                        sid_to_items,
                        mapper_top_k=mapper_k,
                    )
                    item_covered = target_item in ranked_items
                    global_covered = true_global_sid in global_candidates
                    candidate_rates[key].add(item_covered, seen=seen)
                    global_candidate_rates[key].add(global_covered, seen=seen)
                    conditional_candidate_rates[key].add(
                        item_covered,
                        seen=seen,
                        include=event,
                    )
                    conditional_global_candidate_rates[key].add(
                        global_covered,
                        seen=seen,
                        include=event,
                    )
                    conditional_teacher_rates[key].add(
                        teacher_by_k[mapper_k],
                        seen=seen,
                        include=event,
                    )
                    not_event_candidate_rates[key].add(
                        item_covered,
                        seen=seen,
                        include=not event,
                    )
                    not_event_global_candidate_rates[key].add(
                        global_covered,
                        seen=seen,
                        include=not event,
                    )
                    not_event_teacher_rates[key].add(
                        teacher_by_k[mapper_k],
                        seen=seen,
                        include=not event,
                    )
                    joint_gate_rates[key].add(
                        event and teacher_by_k[mapper_k],
                        seen=seen,
                    )
                    candidate_sizes[key].add(len(ranked_items), seen=seen)
                    for recommendation_k in recommendation_cutoffs:
                        final_key = (beam, mapper_k, recommendation_k)
                        final_hit = target_item in ranked_items[:recommendation_k]
                        final_rates[final_key].add(final_hit, seen=seen)
                        conditional_final_rates[final_key].add(
                            final_hit,
                            seen=seen,
                            include=event,
                        )
                        not_event_final_rates[final_key].add(
                            final_hit,
                            seen=seen,
                            include=not event,
                        )
                        gated_final_rates[final_key].add(
                            final_hit,
                            seen=seen,
                            include=event and teacher_by_k[mapper_k],
                        )

    if not example_counts["all"]:
        raise ValueError("mapper diagnostic loader must not be empty")

    grid = {}
    for beam in beam_cutoffs:
        beam_result = {}
        for mapper_k in mapper_cutoffs:
            key = (beam, mapper_k)
            beam_result[str(mapper_k)] = {
                "item_candidate_coverage_before_truncation": (
                    candidate_rates[key].result()
                ),
                "global_sid_candidate_coverage_before_truncation": (
                    global_candidate_rates[key].result()
                ),
                "candidate_count_before_truncation": candidate_sizes[key].result(),
                "joint_E_B_and_teacher_forced_mapper_hit": (
                    joint_gate_rates[key].result()
                ),
                "conditional_on_E_B": {
                    "teacher_forced_mapper_hit": (
                        conditional_teacher_rates[key].result()
                    ),
                    "item_candidate_coverage_before_truncation": (
                        conditional_candidate_rates[key].result()
                    ),
                    "global_sid_candidate_coverage_before_truncation": (
                        conditional_global_candidate_rates[key].result()
                    ),
                    "final_recall": {
                        str(recommendation_k): conditional_final_rates[
                            (beam, mapper_k, recommendation_k)
                        ].result()
                        for recommendation_k in recommendation_cutoffs
                    },
                    "ordering_loss": {
                        str(recommendation_k): _rate_difference(
                            conditional_candidate_rates[key],
                            conditional_final_rates[(beam, mapper_k, recommendation_k)],
                        )
                        for recommendation_k in recommendation_cutoffs
                    },
                },
                "conditional_on_not_E_B": {
                    "teacher_forced_mapper_hit": (
                        not_event_teacher_rates[key].result()
                    ),
                    "item_candidate_coverage_before_truncation": (
                        not_event_candidate_rates[key].result()
                    ),
                    "global_sid_candidate_coverage_before_truncation": (
                        not_event_global_candidate_rates[key].result()
                    ),
                    "final_recall": {
                        str(recommendation_k): not_event_final_rates[
                            (beam, mapper_k, recommendation_k)
                        ].result()
                        for recommendation_k in recommendation_cutoffs
                    },
                    "ordering_loss": {
                        str(recommendation_k): _rate_difference(
                            not_event_candidate_rates[key],
                            not_event_final_rates[(beam, mapper_k, recommendation_k)],
                        )
                        for recommendation_k in recommendation_cutoffs
                    },
                },
                "conditional_on_E_B_and_teacher_forced_mapper_hit": {
                    "final_recall": {
                        str(recommendation_k): gated_final_rates[
                            (beam, mapper_k, recommendation_k)
                        ].result()
                        for recommendation_k in recommendation_cutoffs
                    }
                },
                "final_recall": {
                    str(recommendation_k): final_rates[
                        (beam, mapper_k, recommendation_k)
                    ].result()
                    for recommendation_k in recommendation_cutoffs
                },
                "ordering_loss": {
                    str(recommendation_k): _rate_difference(
                        candidate_rates[key],
                        final_rates[(beam, mapper_k, recommendation_k)],
                    )
                    for recommendation_k in recommendation_cutoffs
                },
            }
        grid[str(beam)] = beam_result

    return {
        "config": {
            "beam_cutoffs": list(beam_cutoffs),
            "mapper_cutoffs": list(mapper_cutoffs),
            "recommendation_cutoffs": list(recommendation_cutoffs),
            "ordering_loss": "candidate_coverage-final_recall@N",
        },
        "examples": example_counts,
        "effective_generator_beams": {
            "min": effective_beam_min or 0,
            "max": effective_beam_max,
        },
        "E_B": {str(beam): event_rates[beam].result() for beam in beam_cutoffs},
        "teacher_forced_mapper_top_k": {
            str(mapper_k): teacher_rates[mapper_k].result()
            for mapper_k in mapper_cutoffs
        },
        "grid": grid,
    }
