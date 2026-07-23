from dataclasses import dataclass

@dataclass
class RQKMeansConfig:
    num_levels: int = 3
    codebook_size: int = 512
    random_state: int = 42
    kmeans_batch_size: int = 1024 * 16
    predict_batch_size: int = 65536