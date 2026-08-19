"""Контрольная точка старого poster-протокола"""

import os
from pathlib import Path

from sids_rank_vs_generate.data import POSTER_FULL_33148_PROTOCOL_ID
from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline
from sids_rank_vs_generate.tracking import MLflowConfig


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size="50m",
        history_days=None,
        protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
    ),
    catalog_from_fit=False,
    reports_dir=Path(
        os.environ.get("SID_REPORTS_DIR", "reports/experiments")
    ),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256)
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=16,
            resolve_collisions=False,
            max_validation_examples=None,
            max_test_examples=None,
            use_bf16=False,
        )
    ),
    tracking=MLflowConfig(
        experiment_name="sid-quality-poster-full-33148",
        run_name="bridge-rqkmeans-l3-c256",
        tags=(
            ("poster", "true"),
            ("sid_model", "rqkmeans"),
            ("model", "rqkmeans"),
            ("representation", "original"),
            ("init", "rqkmeans"),
            ("assignment", "nearest"),
            ("collision_resolution", "none"),
            ("num_levels", "3"),
            ("codebook_size", "256"),
            ("seed", "42"),
            ("dataset", "50m"),
            ("protocol_id", POSTER_FULL_33148_PROTOCOL_ID),
            ("data_scope", "full"),
            ("evaluation_split", "legacy_heldout_reused"),
            ("downstream_recipe", "legacy_global_item_recall_selected"),
            ("expected_catalog_size", "33148"),
            ("branch_base_main_commit", "852f30c"),
            ("verified_compatible_main_commit", "3e0c096"),
            ("series", "bridge"),
            ("variant", "rqkmeans-3x256"),
        ),
    ),
)
