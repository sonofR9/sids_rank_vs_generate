from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.sids.balance_rq_kmeans_pipeline import (
    BalanceRQKMeansPipeline,
)
from sids_rank_vs_generate.sids.config import (
    BalanceRQKMeansConfig,
    PlumRQVAEConfig,
)
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import PlumRQVAEPipeline
from sids_rank_vs_generate.tracking import MLflowConfig


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _codebook_sizes() -> tuple[int, ...]:
    raw = os.environ.get("CODEBOOK_SIZES", "512,256,128")
    try:
        sizes = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("CODEBOOK_SIZES must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("CODEBOOK_SIZES must contain positive integers")
    return sizes


def build_experiment() -> ExperimentConfig:
    sid_model = os.environ.get("SID_MODEL", "balance").strip().lower()
    collision_resolution = os.environ.get(
        "COLLISION_RESOLUTION", "none"
    ).strip().lower()
    if collision_resolution != "none":
        raise ValueError("poster ranking matrix supports COLLISION_RESOLUTION=none only")

    if sid_model == "balance":
        num_levels = _positive_int("NUM_LEVELS", 3)
        codebook_size = _positive_int("CODEBOOK_SIZE", 512)
        pipeline = BalanceRQKMeansPipeline(
            BalanceRQKMeansConfig(
                num_levels=num_levels,
                codebook_size=codebook_size,
                device="auto",
            )
        )
        size_tag = str(codebook_size)
    elif sid_model == "plum":
        sizes = _codebook_sizes()
        pipeline = PlumRQVAEPipeline(
            PlumRQVAEConfig(
                num_levels=len(sizes),
                codebook_size=max(sizes),
                codebook_sizes=sizes,
                device="auto",
            )
        )
        size_tag = "x".join(str(size) for size in sizes)
    else:
        raise ValueError("SID_MODEL must be 'balance' or 'plum'")

    return ExperimentConfig(
        dataset=YambdaDatasetConfig(size="50m"),
        sid_pipeline=pipeline,
        training=TrainingConfig(
            generation=GenerationConfig(resolve_collisions=False),
        ),
        tracking=MLflowConfig(
            tags=(
                ("poster", "true"),
                ("sid_model", sid_model),
                ("collision_resolution", collision_resolution),
                ("codebook_sizes", size_tag),
                ("seed", "42"),
                ("dataset", "50m"),
                ("data_scope", "full"),
                ("evaluation_scope", "ranking_only"),
            )
        ),
        reports_dir=Path("reports/poster_runs"),
    )


EXPERIMENT = build_experiment()
