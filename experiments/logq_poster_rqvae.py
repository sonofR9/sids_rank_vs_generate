"""Environment-driven RQ-VAE generation experiment with enabled LogQ."""

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


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


LOGQ_MODE = os.environ.get("LOGQ_MODE", "corrected").strip().lower()
if LOGQ_MODE not in {"standard", "corrected"}:
    raise ValueError("LOGQ_MODE must be 'standard' or 'corrected'")

DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
NUM_LEVELS = positive_int("NUM_LEVELS", 3)
CODEBOOK_SIZE = positive_int("CODEBOOK_SIZE", 256)
SEED = 42
AUXILIARY_ITEM_LOSS_WEIGHT = float(
    os.environ.get("AUXILIARY_ITEM_LOSS_WEIGHT", "0.1")
)
MODEL_CACHE_DIR = Path(
    os.environ.get("MODEL_CACHE_DIR", "reports/logq_rqvae_runs/_cache")
)

run_name = (
    "logq-rqvae-rqkmeans-nearest-token-"
    f"l{NUM_LEVELS}-c{CODEBOOK_SIZE}-aux0p1-{LOGQ_MODE}-full-seed{SEED}"
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
            psi_top_k=min(4, CODEBOOK_SIZE),
            checkpoint_path=(
                MODEL_CACHE_DIR
                / f"rqvae-{DATASET_SIZE}-full-l{NUM_LEVELS}-c{CODEBOOK_SIZE}-seed{SEED}.pt"
            ),
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
            ("logq_eps", "1e-12"),
            ("logq_debug", "true"),
            ("sid_model", "rqvae"),
            ("model", "rqvae"),
            ("init", "rqkmeans"),
            ("assignment", "nearest"),
            ("collision_resolution", "token"),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("num_levels", str(NUM_LEVELS)),
            ("seed", str(SEED)),
            ("dataset", DATASET_SIZE),
            ("data_scope", "full"),
            ("logq_auxiliary_item_loss_weight", str(AUXILIARY_ITEM_LOSS_WEIGHT)),
        ),
    ),
    reports_dir=Path("reports/logq_rqvae_runs"),
)
