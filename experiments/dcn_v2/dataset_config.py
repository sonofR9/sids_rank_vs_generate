from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_type: str = 'flat'
    dataset_size: str = '50m'
    interaction_name: str = 'multi_event'
    default_like_window_seconds: int = 24 * 60 * 60
    lag_seconds: int = 15 * 60