"""Текущий user-aware v1 для сравнения на свежем main"""

import os
from pathlib import Path

from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import (
    RQKMeansConfig,
    RQKMeansPipeline,
    UserAwareRQKMeansConfig,
    UserAwareRQKMeansPipeline,
    UserSidMapperConfig,
)
from sids_rank_vs_generate.tracking import MLflowConfig


REPORTS_DIR = os.environ.get("SID_REPORTS_DIR")


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m", validation_fraction=0.1),
    catalog_from_fit=True,
    reports_dir=(
        Path(REPORTS_DIR) if REPORTS_DIR else Path("reports/experiments")
    ),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256)
    ),
    user_sid_pipeline=UserAwareRQKMeansPipeline(
        UserAwareRQKMeansConfig(num_levels=3, codebook_size=256)
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=64,
            batch_size=1024,
            batch_size_val=128,
            num_workers=4,
            max_validation_examples=10_000,
            max_test_examples=10_000,
            use_user_profile=True,
            initialize_item_embeddings_from_content=True,
            content_embedding_scale=128**0.5,
            auxiliary_item_loss_weight=0.1,
            sid_only_validation_during_training=True,
            validation_metric="sid_beam_recall@100",
            mapper_top_k=3,
            run_diagnostics=True,
            use_bf16=False,
            mapper=UserSidMapperConfig(
                kind="transformer",
                d_model=192,
                d_ff=512,
                num_heads=6,
                num_encoder_layers=3,
                num_decoder_layers=3,
                batch_size=512,
                inference_batch_size=512,
                lr=3e-4,
                max_epochs=30,
                early_stopping_patience=4,
                early_stopping_min_delta=0.0,
                catalog_masked_teacher_forced_loss=True,
                collision_level_loss_weight=0.5,
                validation_selection_metric=(
                    "autoregressive_catalog_top_k_accuracy"
                ),
                validation_selection_top_k=3,
                decode_beam_size=16,
            ),
        )
    ),
    tracking=MLflowConfig(
        experiment_name="sids-user-aware2-fresh-main",
        tags=(
            ("comparison", "user-aware2.0"),
            ("main_commit", "852f30c"),
            ("variant", "user-aware-v1"),
        ),
    ),
)
