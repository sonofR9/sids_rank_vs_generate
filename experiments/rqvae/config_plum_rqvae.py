from dataclasses import dataclass

@dataclass
class PLUMRQVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_sizes: tuple[int, ...] = (2048, 1024, 512)
    codebook_size: int = 2048
    batch_size: int = 1024
    num_epochs: int = 50
    lr: float = 1e-3
    beta: float = 0.25
    num_workers: int = 4
    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3
