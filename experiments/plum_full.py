from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import PlumRQVAEPipeline
from sids_rank_vs_generate.sids.config import PlumRQVAEConfig


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=PlumRQVAEPipeline(
        PlumRQVAEConfig(
            codebook_sizes=(512, 256),
            device="auto",
        )
    ),
)
