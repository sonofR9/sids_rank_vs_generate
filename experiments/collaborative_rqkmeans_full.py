"""OneRec-style collaborative-aware RQ-KMeans ranking/generation experiment."""

from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import (
    CollaborativeAlignmentConfig,
    CollaborativeAwareRQKMeansConfig,
    CollaborativeAwareRQKMeansPipeline,
)


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=CollaborativeAwareRQKMeansPipeline(
        CollaborativeAwareRQKMeansConfig(
            num_levels=3,
            codebook_size=256,
            alignment=CollaborativeAlignmentConfig(
                history_window=3,
                min_pair_support=2,
                pairs_per_item=20,
                max_pairs=1_000_000,
                hidden_dim=512,
                batch_size=1024,
                epochs=10,
                learning_rate=1e-3,
                temperature=0.07,
                content_preservation_weight=0.1,
            ),
        )
    ),
)
