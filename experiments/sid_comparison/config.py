# единое место, чтобы работать со всеми конфигами
from experiments.config_semantic_transformer import (
    SemanticTransformerConfig,
)
from experiments.rqkmeans.config_rqkmeans import RQKMeansConfig
from experiments.rqvae.config_rqvae import RQVAEConfig

SID_Methods = ("rqkmeans", "rqvae")

SID_Hyperparametrs = (
    {"num_levels": 3, "codebook_size": 256},
)

def make_transformer_config() -> SemanticTransformerConfig:
    cfg = SemanticTransformerConfig()

    cfg.count_of_samples = 5_000_000
    cfg.num_beams = 150
    cfg.recall_k = 100

    cfg.max_epochs = 30
    cfg.early_stopping_patience = 3
    cfg.early_stopping_min_delta = 1e-4

    return cfg

def make_rqkmeans_config(
    num_levels: int,
    codebook_size: int,
) -> RQKMeansConfig:
    cfg = RQKMeansConfig()

    cfg.num_levels = num_levels
    cfg.codebook_size = codebook_size

    return cfg

def make_rqvae_config(
    num_levels: int,
    codebook_size: int,
) -> RQVAEConfig:
    cfg = RQVAEConfig()

    cfg.num_levels = num_levels
    cfg.codebook_size = codebook_size

    return cfg