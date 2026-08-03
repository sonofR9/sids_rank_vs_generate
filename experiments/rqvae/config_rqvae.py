from dataclasses import dataclass

@dataclass
class RQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 512
    codebook_sizes: tuple[int, ...] | None = None
    batch_size: int = 1024
    num_epochs: int = 50
    lr: float = 1e-3
    beta: float = 0.25
    num_workers: int = 4
