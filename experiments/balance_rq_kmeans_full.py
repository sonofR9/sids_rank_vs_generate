from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids.r3vae import BalanceRQKMeansPipeline
from sids_rank_vs_generate.sids.config import BalanceRQKMeansConfig


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=BalanceRQKMeansPipeline(
        BalanceRQKMeansConfig(
            num_levels = 3,
            codebook_size = 512,
            device="auto",
        )
    ),
)
