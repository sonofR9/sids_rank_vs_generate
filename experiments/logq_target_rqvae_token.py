"""Controlled LogQ experiment on the strongest poster SID configuration."""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import RQVAEConfig, RQVAEPipeline
from sids_rank_vs_generate.tracking import MLflowConfig


LOGQ_MODE = os.environ.get("LOGQ_MODE", "corrected").strip().lower()
if LOGQ_MODE not in {"none", "corrected"}:
    raise ValueError("LOGQ_MODE must be 'none' or 'corrected'")

DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
AUXILIARY_ITEM_LOSS_WEIGHT = 0.1
NUM_LEVELS = 2
CODEBOOK_SIZE = 256
SEED = 42

mode_label = "baseline" if LOGQ_MODE == "none" else "corrected"
run_name = (
    "logq-target-rqvae-rqkmeans-nearest-token-"
    f"l{NUM_LEVELS}-c{CODEBOOK_SIZE}-aux0p1-{mode_label}-full-seed{SEED}"
)

EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size=DATASET_SIZE),
    sid_pipeline=RQVAEPipeline(
        RQVAEConfig(
            hidden_dim=256,
            latent_dim=128,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
            batch_size=1_024,
            lr=5e-4,
            pretrain_epochs=3,
            max_epochs=20,
            assignment_method="nearest",
            psi_top_k=4,
        )
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            resolve_collisions=True,
            auxiliary_item_loss_weight=AUXILIARY_ITEM_LOSS_WEIGHT,
            logq_mode=LOGQ_MODE,
            logq_debug=True,
            gather_distributed_negatives=False,
        )
    ),
    tracking=MLflowConfig(
        run_name=run_name,
        tags=(
            ("logq_target", "true"),
            ("logq_mode", LOGQ_MODE),
            ("sid_model", "rqvae"),
            ("init", "rqkmeans"),
            ("assignment", "nearest"),
            ("collision_resolution", "token"),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("num_levels", str(NUM_LEVELS)),
            ("seed", str(SEED)),
            ("dataset", DATASET_SIZE),
            ("data_scope", "full"),
        ),
    ),
    reports_dir=Path("reports/logq_target_runs"),
)
