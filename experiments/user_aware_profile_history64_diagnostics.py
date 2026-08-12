"""Диагностика mapper на готовых весах"""

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
MAPPER_CHECKPOINT = os.environ.get("SID_MAPPER_CHECKPOINT")
REFERENCE_RUN_DIR = os.environ.get("SID_REFERENCE_RUN_DIR")
REPORTS_DIR = os.environ.get("SID_REPORTS_DIR")
if not GENERATOR_CHECKPOINT or not MAPPER_CHECKPOINT or not REFERENCE_RUN_DIR:
    raise RuntimeError(
        "SID_GENERATOR_CHECKPOINT, SID_MAPPER_CHECKPOINT and "
        "SID_REFERENCE_RUN_DIR are required"
    )


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
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
            use_user_profile=True,
            initialize_item_embeddings_from_content=True,
            content_embedding_scale=128**0.5,
            auxiliary_item_loss_weight=0.1,
            sid_only_validation_during_training=True,
            validation_metric="sid_beam_recall@150",
            run_diagnostics=True,
            run_mapper_inference_diagnostic=True,
            load_checkpoint_path=Path(GENERATOR_CHECKPOINT),
            reference_artifact_dir=Path(REFERENCE_RUN_DIR),
            mapper=UserSidMapperConfig(
                kind="transformer",
                load_checkpoint_path=Path(MAPPER_CHECKPOINT),
            ),
        )
    ),
)
