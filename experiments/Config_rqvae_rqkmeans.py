from dataclasses import dataclass

@dataclass
class RQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 512
    batch_size: int = 1024
    num_epochs: int = 50
    lr: float = 1e-3
    beta: float = 0.25
    num_workers: int = 4


@dataclass
class RQKMeansConfig:
    num_levels: int = 3
    codebook_size: int = 512
    random_state: int = 42
    kmeans_batch_size: int = 1024 * 16
    predict_batch_size: int = 65536