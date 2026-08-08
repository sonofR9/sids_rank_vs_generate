"""TransAct на обычных item id без семантических признаков"""

from sids_rank_vs_generate.evaluation import RankingConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    catalog_from_fit=True,
    sid_pipeline=RQKMeansPipeline(RQKMeansConfig(num_levels=3, codebook_size=256)),
    training=TrainingConfig(
        ranking=RankingConfig(
            models=("transact",),
            use_semantic_ids=False,
            num_workers=8,
            prefetch_factor=4,
        )
    ),
)
