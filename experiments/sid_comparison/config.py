# единое место, чтобы работать со всеми конфигами
from experiments.config_semantic_transformer import (
    SemanticTransformerConfig,
)
SID_Methods = ("plum",)

COLLISION_RESOLUTION_MODES = (False, True)

MLFLOW_EXPERIMENT_NAME = "sid-comparison"

MLFLOW_TRACKING_URI = "http://51.250.47.231:5000"

SID_Hyperparametrs = (
    {"codebook_sizes": (256, 128, 64)},
    {"codebook_sizes": (512, 256, 128)},
    {"codebook_sizes": (256, 128, 64, 32)},
    {"codebook_sizes": (1024, 512)},
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
