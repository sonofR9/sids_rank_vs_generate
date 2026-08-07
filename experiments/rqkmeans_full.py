"""RQ-KMeans evaluated in ranking and TIGER-like generation."""

from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    catalog_from_fit=True,
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(
            num_levels=3,
            codebook_size=256,
        )
    ),
)
