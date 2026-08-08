from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids.r3vae import R3VAEConfig, R3VAESid


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=R3VAESid(
        R3VAEConfig(
            codebook_sizes=(256, 256, 256),
            epochs=20,
            batch_size=1_024,
            learning_rate=5e-4,
            seed=42,
            device="auto",
        )
    ),
)
