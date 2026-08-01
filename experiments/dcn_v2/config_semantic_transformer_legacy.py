from dataclasses import dataclass

@dataclass
class SemanticTransformerConfig:
    max_history_len: int = 16

    d_model: int = 128
    d_ff: int = 512
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    num_levels: int = 3
    codebook_size: int = 512
