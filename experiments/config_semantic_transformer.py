from dataclasses import dataclass

import torch
from pathlib import Path


@dataclass
class SemanticTransformerConfig:
    count_of_samples: int = 5_000_000
    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    max_epochs: int = 30
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    max_history_len: int = 16
    min_history_len: int = 5
    seconds_per_day: int = 86_400
    val_num_days: int = 1

    d_model: int = 128
    d_ff: int = 512
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    batch_size: int = 1024
    batch_size_val: int = 64
    num_workers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-2
    num_epochs: int = 5
    grad_clip: float = 1.0

    rq_num_levels: int = 3
    num_levels: int = 4
    codebook_size: int = 512
    codebook_sizes: list[int] | None = None    
    num_beams: int = 150
    recall_k: int = 100

    log_every_n_steps: int = 100
    val_subset_size: int | None = 50_000
    resolve_collisions: bool = True
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )