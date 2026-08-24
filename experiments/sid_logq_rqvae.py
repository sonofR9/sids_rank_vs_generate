"""Pure SID-level LogQ experiment without an auxiliary item loss."""

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


SID_LOGQ_MODE = os.environ.get("SID_LOGQ_MODE", "standard").strip().lower()
if SID_LOGQ_MODE not in {"none", "standard"}:
    raise ValueError("SID_LOGQ_MODE must be 'none' or 'standard'")

DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
SID_LOGQ_BETA = float(os.environ.get("SID_LOGQ_BETA", "1.0"))
if SID_LOGQ_BETA <= 0:
    raise ValueError("SID_LOGQ_BETA must be positive")

NUM_LEVELS = int(os.environ.get("NUM_LEVELS", "2"))
CODEBOOK_SIZE = int(os.environ.get("CODEBOOK_SIZE", "256"))
SEED = 42
mode_label = "baseline" if SID_LOGQ_MODE == "none" else "logq"
beta_label = str(SID_LOGQ_BETA).replace(".", "p")
run_name = (
    "sid-logq-rqvae-rqkmeans-nearest-token-"
    f"l{NUM_LEVELS}-c{CODEBOOK_SIZE}-aux0-{mode_label}-"
    f"beta{beta_label}-full-seed{SEED}"
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
        )
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            resolve_collisions=True,
            auxiliary_item_loss_weight=0.0,
            logq_mode="none",
            sid_logq_mode=SID_LOGQ_MODE,
            sid_logq_beta=SID_LOGQ_BETA,
            gather_distributed_negatives=False,
        )
    ),
    tracking=MLflowConfig(
        run_name=run_name,
        tags=(
            ("sid_logq_target", "true"),
            ("sid_logq_mode", SID_LOGQ_MODE),
            ("sid_logq_beta", str(SID_LOGQ_BETA)),
            ("sid_logq_alpha", "1.0_fixed"),
            ("auxiliary_item_loss_weight", "0.0"),
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
        ),
    ),
    reports_dir=Path("reports/sid_logq_runs"),
)
