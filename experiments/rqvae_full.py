"""Контентные SID через RQ-VAE"""

from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQVAEConfig, RQVAEPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    catalog_from_fit=True,
    sid_pipeline=RQVAEPipeline(RQVAEConfig(num_levels=3, codebook_size=256)),
)
