from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import RQVAELetterPipeline
from sids_rank_vs_generate.sids.config import RQVAELetterConfig
EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=RQVAELetterPipeline(
        RQVAELetterConfig(
            latent_dim=128,
            collaborative_embedding_dim=128,
        )
    ),
)
