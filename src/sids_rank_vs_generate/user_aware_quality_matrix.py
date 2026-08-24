from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from io import StringIO
from typing import Any, Iterable, Sequence

MATRIX_ID = "user-aware-quality-30-current-anchor-v1"
# Дублируется намеренно: dry-run не должен требовать ML зависимости
PROTOCOL_ID = "poster-full-33148-v1"
PROTOCOL_TAG = "legacy_poster_full_33148"
QUALITY_METRIC_SCHEMA = "sid-quality-v1"
EXPECTED_CATALOG_SIZE = 33_148
EXPERIMENT_NAME = (
    "sid-quality-user-aware-current-anchor-poster-full-33148"
)
BRANCH_BASE_MAIN_COMMIT = "852f30c"
VERIFIED_COMPATIBLE_MAIN_COMMIT = "3e0c096"
GLOBAL_SID_FAMILY = "rq-kmeans"
GLOBAL_SID_CODEBOOK_SIZES = (256, 256, 256)
GLOBAL_SID_ANCHOR_ID = "current-rqkmeans-l3-c256-seed42-v1"
GLOBAL_SID_REFERENCE_SCOPE = "current_anchor"
LEGACY_POSTER_REFERENCE_MATCH = "not_exact"
EXPERIMENT_SEED = 42

V1_FAMILY = "user-aware-v1"
V2_FAMILY = "user-aware-v2"
FAMILIES = (V1_FAMILY, V2_FAMILY)

CAPACITY_DEPTH_SERIES = "capacity-depth"
CAPACITY_WIDTH_SERIES = "capacity-width"
USER_WEIGHT_SERIES = "user-weight"
TAIL_CODEBOOK_SERIES = "tail-codebook"
PREFIX_PERMUTATION_SERIES = "prefix-permutation"
SERIES = (
    CAPACITY_DEPTH_SERIES,
    CAPACITY_WIDTH_SERIES,
    USER_WEIGHT_SERIES,
    TAIL_CODEBOOK_SERIES,
    PREFIX_PERMUTATION_SERIES,
)

BASE_CODEBOOK_SIZES = (256, 256, 256)
V1_BASE_USER_PROFILE_WEIGHT = 1.0
V2_BASE_USER_PROFILE_WEIGHT = 1.0
V2_BASE_FUSION_WEIGHT = 0.35


def _float_slug(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _permutation_text(permutation: Sequence[int]) -> str:
    return "".join(str(level) for level in permutation)


@dataclass(frozen=True)
class UserAwareQualityRun:
    """Одна точка будущей user-aware серии"""

    run_name: str
    family: str
    series: tuple[str, ...]
    personal_codebook_sizes: tuple[int, ...]
    user_profile_weight: float
    fusion_weight: float | None
    level_permutation: tuple[int, ...]
    seed: int = EXPERIMENT_SEED

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        if not self.run_name or any(character.isspace() for character in self.run_name):
            raise ValueError("run_name must be non-empty and contain no whitespace")
        if not self.series or any(name not in SERIES for name in self.series):
            raise ValueError("series must contain known matrix series")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series must not contain duplicates")
        if not self.personal_codebook_sizes or any(
            size <= 0 for size in self.personal_codebook_sizes
        ):
            raise ValueError("personal_codebook_sizes must contain positive values")
        if self.user_profile_weight < 0:
            raise ValueError("user_profile_weight must be non-negative")
        if self.family == V1_FAMILY and self.fusion_weight is not None:
            raise ValueError("v1 must not define fusion_weight")
        if self.family == V2_FAMILY and self.fusion_weight is None:
            raise ValueError("v2 must define fusion_weight")
        if self.fusion_weight is not None and not 0 <= self.fusion_weight <= 1:
            raise ValueError("fusion_weight must be in [0, 1]")
        expected_levels = tuple(range(len(self.personal_codebook_sizes)))
        if tuple(sorted(self.level_permutation)) != expected_levels:
            raise ValueError("level_permutation must contain every level exactly once")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def signature(self) -> tuple[Any, ...]:
        """Параметры уникальной точки"""

        return (
            self.family,
            self.personal_codebook_sizes,
            self.user_profile_weight,
            self.fusion_weight,
            self.level_permutation,
            self.seed,
        )

    @property
    def tags(self) -> dict[str, str]:
        """Стабильные теги запуска"""

        return {
            "matrix_id": MATRIX_ID,
            "matrix_run_id": self.run_name,
            "protocol_id": PROTOCOL_ID,
            "protocol": PROTOCOL_TAG,
            "quality_metric_schema": QUALITY_METRIC_SCHEMA,
            "expected_catalog_size": str(EXPECTED_CATALOG_SIZE),
            "branch_base_main_commit": BRANCH_BASE_MAIN_COMMIT,
            "verified_compatible_main_commit": (
                VERIFIED_COMPATIBLE_MAIN_COMMIT
            ),
            "data_scope": "full",
            "evaluation_split": "legacy_heldout_reused",
            "sid_scope": "personalized",
            "x_scope": "user_catalog_macro",
            "quality_aggregation": "per_user_catalog_macro",
            "downstream_recipe": "personalized_user_conditioned_sid_selected",
            "pooled_global_correlation": "disallowed",
            "sid_family": self.family,
            "matrix_series": ",".join(self.series),
            "global_sid_family": GLOBAL_SID_FAMILY,
            "global_sid_anchor_id": GLOBAL_SID_ANCHOR_ID,
            "global_sid_reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
            "legacy_poster_reference_match": (
                LEGACY_POSTER_REFERENCE_MATCH
            ),
            "global_sid_codebook_sizes": ",".join(
                str(size) for size in GLOBAL_SID_CODEBOOK_SIZES
            ),
            "personal_sid_codebook_sizes": ",".join(
                str(size) for size in self.personal_codebook_sizes
            ),
            "user_profile_weight": format(self.user_profile_weight, "g"),
            "fusion_weight": (
                "none"
                if self.fusion_weight is None
                else format(self.fusion_weight, "g")
            ),
            "level_permutation": _permutation_text(self.level_permutation),
            "seed": str(self.seed),
            "downstream_metric": "final_item_recall_at_100",
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимое описание запуска"""

        return {
            "run_name": self.run_name,
            "experiment_name": EXPERIMENT_NAME,
            "family": self.family,
            "series": list(self.series),
            "personal_sid": {
                "codebook_sizes": list(self.personal_codebook_sizes),
                "user_profile_weight": self.user_profile_weight,
                "fusion_weight": self.fusion_weight,
                "level_permutation": list(self.level_permutation),
            },
            "seed": self.seed,
            "tags": self.tags,
        }


def _run_name(
    family: str,
    codebook_sizes: tuple[int, ...],
    user_profile_weight: float,
    fusion_weight: float | None,
    level_permutation: tuple[int, ...],
) -> str:
    prefix = "uaq30-v1" if family == V1_FAMILY else "uaq30-v2"
    identity = tuple(range(len(codebook_sizes)))
    if level_permutation != identity:
        return f"{prefix}-prefix-permutation-{_permutation_text(level_permutation)}"
    if codebook_sizes != BASE_CODEBOOK_SIZES:
        if len(set(codebook_sizes)) == 1:
            return (
                f"{prefix}-capacity-l{len(codebook_sizes)}"
                f"-c{codebook_sizes[0]}"
            )
        if len(codebook_sizes) == 3 and codebook_sizes[:2] == (256, 256):
            return f"{prefix}-tail-k3-{codebook_sizes[2]}"
        sizes = "x".join(str(size) for size in codebook_sizes)
        return f"{prefix}-capacity-{sizes}"
    if family == V1_FAMILY and user_profile_weight != V1_BASE_USER_PROFILE_WEIGHT:
        return f"{prefix}-user-weight-{_float_slug(user_profile_weight)}"
    if family == V2_FAMILY and fusion_weight != V2_BASE_FUSION_WEIGHT:
        assert fusion_weight is not None
        return f"{prefix}-fusion-weight-{_float_slug(fusion_weight)}"
    return f"{prefix}-anchor-l3-c256"


def _candidate(
    family: str,
    series: str,
    *,
    codebook_sizes: tuple[int, ...] = BASE_CODEBOOK_SIZES,
    user_profile_weight: float = V1_BASE_USER_PROFILE_WEIGHT,
    fusion_weight: float | None = None,
    level_permutation: tuple[int, ...] | None = None,
) -> UserAwareQualityRun:
    if family == V2_FAMILY and fusion_weight is None:
        fusion_weight = V2_BASE_FUSION_WEIGHT
    permutation = level_permutation or tuple(range(len(codebook_sizes)))
    return UserAwareQualityRun(
        run_name=_run_name(
            family,
            codebook_sizes,
            user_profile_weight,
            fusion_weight,
            permutation,
        ),
        family=family,
        series=(series,),
        personal_codebook_sizes=codebook_sizes,
        user_profile_weight=user_profile_weight,
        fusion_weight=fusion_weight,
        level_permutation=permutation,
    )


def _candidates_for_family(family: str) -> Iterable[UserAwareQualityRun]:
    for levels in (2, 3, 4):
        yield _candidate(
            family,
            CAPACITY_DEPTH_SERIES,
            codebook_sizes=(256,) * levels,
        )
    for width in (128, 256, 512, 1024):
        yield _candidate(
            family,
            CAPACITY_WIDTH_SERIES,
            codebook_sizes=(width,) * 3,
        )
    if family == V1_FAMILY:
        for weight in (0.0, 0.25, 0.5, 1.0, 2.0):
            yield _candidate(
                family,
                USER_WEIGHT_SERIES,
                user_profile_weight=weight,
            )
    else:
        for weight in (0.0, 0.2, 0.35, 0.7, 1.0):
            yield _candidate(
                family,
                USER_WEIGHT_SERIES,
                fusion_weight=weight,
            )
    for tail_size in (32, 64, 128, 256):
        yield _candidate(
            family,
            TAIL_CODEBOOK_SERIES,
            codebook_sizes=(256, 256, tail_size),
        )
    for permutation in ((0, 1, 2), (1, 0, 2), (2, 1, 0)):
        yield _candidate(
            family,
            PREFIX_PERMUTATION_SERIES,
            level_permutation=permutation,
        )


def build_user_aware_quality_matrix() -> tuple[UserAwareQualityRun, ...]:
    """Собрать 30 точек и объединить общие anchors"""

    unique: dict[tuple[Any, ...], UserAwareQualityRun] = {}
    for family in FAMILIES:
        for candidate in _candidates_for_family(family):
            previous = unique.get(candidate.signature)
            if previous is None:
                unique[candidate.signature] = candidate
                continue
            merged_series = previous.series + tuple(
                name for name in candidate.series if name not in previous.series
            )
            unique[candidate.signature] = replace(previous, series=merged_series)

    runs = tuple(unique.values())
    if len(runs) != 30:
        raise RuntimeError(f"quality matrix must contain 30 runs, got {len(runs)}")
    run_names = [run.run_name for run in runs]
    if len(set(run_names)) != len(run_names):
        raise RuntimeError("quality matrix run names must be unique")
    return runs


USER_AWARE_QUALITY_MATRIX = build_user_aware_quality_matrix()


def matrix_manifest(
    runs: Sequence[UserAwareQualityRun] = USER_AWARE_QUALITY_MATRIX,
) -> dict[str, Any]:
    """Dry-run manifest без обучения"""

    return {
        "schema_version": 1,
        "dry_run": True,
        "executes_training": False,
        "matrix_id": MATRIX_ID,
        "protocol": {
            "id": PROTOCOL_ID,
            "expected_catalog_size": EXPECTED_CATALOG_SIZE,
        },
        "quality_metric_schema": QUALITY_METRIC_SCHEMA,
        "experiment_name": EXPERIMENT_NAME,
        "fixed_global_sid": {
            "family": GLOBAL_SID_FAMILY,
            "codebook_sizes": list(GLOBAL_SID_CODEBOOK_SIZES),
            "anchor_id": GLOBAL_SID_ANCHOR_ID,
            "reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
            "legacy_poster_reference_match": (
                LEGACY_POSTER_REFERENCE_MATCH
            ),
        },
        "run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
    }


def _table(runs: Sequence[UserAwareQualityRun]) -> str:
    headers = (
        "#",
        "run_name",
        "family",
        "series",
        "personal_codebooks",
        "user_weight",
        "fusion_weight",
        "permutation",
    )
    rows = [
        (
            str(index),
            run.run_name,
            run.family,
            ",".join(run.series),
            "x".join(str(size) for size in run.personal_codebook_sizes),
            format(run.user_profile_weight, "g"),
            "-" if run.fusion_weight is None else format(run.fusion_weight, "g"),
            _permutation_text(run.level_permutation),
        )
        for index, run in enumerate(runs, start=1)
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in rows
    )
    lines.append(
        f"dry-run only: {len(runs)} runs, protocol={PROTOCOL_ID}, "
        f"global_sid_anchor={GLOBAL_SID_ANCHOR_ID}, "
        f"legacy_reference={LEGACY_POSTER_REFERENCE_MATCH}"
    )
    return "\n".join(lines)


def _csv_text(runs: Sequence[UserAwareQualityRun]) -> str:
    output = StringIO()
    fieldnames = (
        "run_name",
        "family",
        "series",
        "personal_codebook_sizes",
        "user_profile_weight",
        "fusion_weight",
        "level_permutation",
        "seed",
        "protocol_id",
        "expected_catalog_size",
        "quality_metric_schema",
        "global_sid_anchor_id",
        "global_sid_reference_scope",
        "legacy_poster_reference_match",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for run in runs:
        writer.writerow(
            {
                "run_name": run.run_name,
                "family": run.family,
                "series": ",".join(run.series),
                "personal_codebook_sizes": ",".join(
                    str(size) for size in run.personal_codebook_sizes
                ),
                "user_profile_weight": format(run.user_profile_weight, "g"),
                "fusion_weight": (
                    ""
                    if run.fusion_weight is None
                    else format(run.fusion_weight, "g")
                ),
                "level_permutation": _permutation_text(run.level_permutation),
                "seed": run.seed,
                "protocol_id": PROTOCOL_ID,
                "expected_catalog_size": EXPECTED_CATALOG_SIZE,
                "quality_metric_schema": QUALITY_METRIC_SCHEMA,
                "global_sid_anchor_id": GLOBAL_SID_ANCHOR_ID,
                "global_sid_reference_scope": GLOBAL_SID_REFERENCE_SCOPE,
                "legacy_poster_reference_match": (
                    LEGACY_POSTER_REFERENCE_MATCH
                ),
            }
        )
    return output.getvalue().rstrip("\r\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Показать 30 user-aware экспериментов "
            "без запуска обучения"
        )
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Формат dry-run manifest",
    )
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        help="Показать только одно семейство",
    )
    parser.add_argument(
        "--series",
        choices=SERIES,
        help="Показать только одну серию",
    )
    parser.add_argument(
        "--run-name",
        help="Показать один будущий запуск",
    )
    return parser


def _select_runs(args: argparse.Namespace) -> tuple[UserAwareQualityRun, ...]:
    runs = USER_AWARE_QUALITY_MATRIX
    if args.family is not None:
        runs = tuple(run for run in runs if run.family == args.family)
    if args.series is not None:
        runs = tuple(run for run in runs if args.series in run.series)
    if args.run_name is not None:
        runs = tuple(run for run in runs if run.run_name == args.run_name)
    if not runs:
        raise ValueError("dry-run filter matched no configurations")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = _select_runs(args)
    if args.format == "json":
        rendered = json.dumps(
            matrix_manifest(runs),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    elif args.format == "csv":
        rendered = _csv_text(runs)
    else:
        rendered = _table(runs)
    sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
