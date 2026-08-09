"""RQ-VAE assignment comparison with fixed training hyperparameters."""

from dataclasses import replace

from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQVAEConfig, RQVAEPipeline


BASE_CONFIG = RQVAEConfig(
    hidden_dim=256,
    latent_dim=128,
    num_levels=3,
    codebook_size=256,
    batch_size=1_024,
    lr=5e-4,
    pretrain_epochs=3,
    max_epochs=20,
    psi_top_k=4,
)


NEAREST = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=RQVAEPipeline(
        replace(BASE_CONFIG, assignment_method="nearest")
    ),
)

RRS = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=RQVAEPipeline(replace(BASE_CONFIG, assignment_method="rrs")),
)

ECM = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=RQVAEPipeline(replace(BASE_CONFIG, assignment_method="ecm")),
)
