"""Mapper v2 с замороженным генератором"""

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


GENERATOR_CHECKPOINT = os.environ.get("SID_GENERATOR_CHECKPOINT")
REFERENCE_RUN_DIR = os.environ.get("SID_REFERENCE_RUN_DIR")
REPORTS_DIR = os.environ.get("SID_REPORTS_DIR")
if not GENERATOR_CHECKPOINT or not REFERENCE_RUN_DIR:
    raise RuntimeError(
        "SID_GENERATOR_CHECKPOINT and SID_REFERENCE_RUN_DIR are required"
    )


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    catalog_from_fit=True,
    reports_dir=(Path(REPORTS_DIR) if REPORTS_DIR else Path("reports/experiments")),
    sid_pipeline=RQKMeansPipeline(RQKMeansConfig(num_levels=3, codebook_size=256)),
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
            use_user_profile=True,
            mapper_top_k=3,
            run_diagnostics=True,
            load_checkpoint_path=Path(GENERATOR_CHECKPOINT),
            reference_artifact_dir=Path(REFERENCE_RUN_DIR),
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
                validation_selection_metric=("autoregressive_catalog_top_k_accuracy"),
                validation_selection_top_k=3,
                decode_beam_size=16,
            ),
        )
    ),
)
