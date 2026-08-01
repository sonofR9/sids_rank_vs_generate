"""RQ-KMeans evaluated in ranking and TIGER-like generation."""

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.evaluation.ranking_stage import RankingConfig
from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size="50m",
        latest_interactions=200_000,
    ),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(
            num_levels=3,
            codebook_size=256,
            random_state=42,
        )
    ),
    ranking=RankingConfig(
        models=("dcnv2", "transact"),
    ),
    generation=GenerationConfig(),
    seed=42,
)
