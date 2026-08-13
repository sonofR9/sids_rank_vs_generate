"""Content-only baseline для сравнения на свежем main"""

import os
from pathlib import Path

from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline
from sids_rank_vs_generate.tracking import MLflowConfig


REPORTS_DIR = os.environ.get("SID_REPORTS_DIR")


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m", validation_fraction=0.1),
    catalog_from_fit=True,
    reports_dir=(
        Path(REPORTS_DIR) if REPORTS_DIR else Path("reports/experiments")
    ),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256)
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=64,
            batch_size=1024,
            batch_size_val=128,
            num_workers=4,
            max_validation_examples=10_000,
            max_test_examples=10_000,
            initialize_item_embeddings_from_content=True,
            content_embedding_scale=128**0.5,
            auxiliary_item_loss_weight=0.1,
            use_bf16=False,
        )
    ),
    tracking=MLflowConfig(
        experiment_name="sids-user-aware2-fresh-main",
        tags=(
            ("comparison", "user-aware2.0"),
            ("main_commit", "852f30c"),
            ("variant", "content-only"),
        ),
    ),
)
