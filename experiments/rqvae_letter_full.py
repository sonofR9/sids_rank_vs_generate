"""LETTER generation experiment across semantic ID sizes."""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import RQVAELetterPipeline
from sids_rank_vs_generate.sids.config import RQVAELetterConfig
from sids_rank_vs_generate.tracking import MLflowConfig


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
NUM_LEVELS = positive_int("NUM_LEVELS", 3)
CODEBOOK_SIZE = positive_int("CODEBOOK_SIZE", 256)

run_name = f"letter-l{NUM_LEVELS}-c{CODEBOOK_SIZE}"


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size=DATASET_SIZE),

    sid_pipeline=RQVAELetterPipeline(
        RQVAELetterConfig(
            latent_dim=128,
            collaborative_embedding_dim=128,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
        )
    ),

    training=TrainingConfig(
        generation=GenerationConfig(
            resolve_collisions=False,
            auxiliary_item_loss_weight=0.0,
            logq_mode="standard",
            logq_debug=False,
            gather_distributed_negatives=False,
        )
    ),

    tracking=MLflowConfig(
        run_name=run_name,
        tags=(
            ("sid_model", "letter"),
            ("model", "letter"),
            ("collision_resolution", "none"),
            ("logq_mode", "standard"),
            ("auxiliary_item_loss_weight", "0.0"),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("num_levels", str(NUM_LEVELS)),
            ("dataset", DATASET_SIZE),
        ),
    ),

    reports_dir=Path("reports/letter_semantic_sizes"),
)
