"""Персональные SID с Transformer маппером"""

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


MAPPER_CHECKPOINT = os.environ.get("SID_MAPPER_CHECKPOINT")


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    catalog_from_fit=True,
    sid_pipeline=RQKMeansPipeline(RQKMeansConfig(num_levels=3, codebook_size=256)),
    user_sid_pipeline=UserAwareRQKMeansPipeline(
        UserAwareRQKMeansConfig(num_levels=3, codebook_size=256)
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            mapper=UserSidMapperConfig(
                kind="transformer",
                load_checkpoint_path=(
                    Path(MAPPER_CHECKPOINT) if MAPPER_CHECKPOINT else None
                ),
            )
        )
    ),
)
