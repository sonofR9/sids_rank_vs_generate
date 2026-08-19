"""Poster-compatible pair-aligned RQ-VAE experiment."""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.data import POSTER_FULL_33148_PROTOCOL_ID
from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import (
    CollaborativeAlignmentConfig,
    CollaborativeAwareRQVAEConfig,
    CollaborativeAwareRQVAEPipeline,
)
from sids_rank_vs_generate.tracking import MLflowConfig


NUM_LEVELS = 3
CODEBOOK_SIZE = 256
SEED = 42


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size="50m",
        history_days=None,
        protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
    ),
    catalog_from_fit=False,
    reports_dir=Path(
        os.environ.get(
            "SID_REPORTS_DIR",
            "reports/poster_pair_aligned_rqvae",
        )
    ),
    sid_pipeline=CollaborativeAwareRQVAEPipeline(
        CollaborativeAwareRQVAEConfig(
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
            alignment=CollaborativeAlignmentConfig(
                history_window=3,
                min_pair_support=2,
                pairs_per_item=20,
                max_pairs=1_000_000,
                hidden_dim=512,
                batch_size=1_024,
                inference_batch_size=65_536,
                epochs=10,
                learning_rate=1e-3,
                weight_decay=1e-4,
                temperature=0.07,
                content_preservation_weight=0.1,
                seed=SEED,
            ),
        )
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=16,
            resolve_collisions=False,
            max_validation_examples=None,
            max_test_examples=None,
            use_bf16=False,
        )
    ),
    tracking=MLflowConfig(
        experiment_name="sid-comparison",
        run_name=(
            "poster-onerec-pair-rqvae-rqkmeans-nearest-none-"
            "l3-c256-full-seed42-catalog33148"
        ),
        tags=(
            ("poster", "true"),
            ("sid_model", "collaborative_rqvae"),
            ("model", "collaborative_rqvae"),
            ("representation", "pair_aligned"),
            ("init", "rqkmeans"),
            ("assignment", "nearest"),
            ("collision_resolution", "none"),
            ("num_levels", str(NUM_LEVELS)),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("seed", str(SEED)),
            ("dataset", "50m"),
            ("protocol_id", POSTER_FULL_33148_PROTOCOL_ID),
            ("data_scope", "full"),
            ("evaluation_split", "legacy_heldout_reused"),
            ("downstream_recipe", "legacy_global_item_recall_selected"),
            ("expected_catalog_size", "33148"),
            ("pair_source", "train_interactions"),
            ("series", "pair-alignment-ablation"),
            ("variant", "pair-aligned-rqvae-3x256"),
        ),
    ),
)
