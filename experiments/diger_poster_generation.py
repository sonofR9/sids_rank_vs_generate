"""DIGER SID в общем poster-протоколе генерации"""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.data import POSTER_FULL_33148_PROTOCOL_ID
from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import PrecomputedSidConfig, PrecomputedSidPipeline
from sids_rank_vs_generate.tracking import MLflowConfig


VARIANT = os.environ.get("DIGER_VARIANT", "").strip().lower()
if VARIANT not in {"both", "frqud"}:
    raise ValueError("DIGER_VARIANT must be both or frqud")

SID_PATH_TEXT = os.environ.get("DIGER_SID_PATH")
if not SID_PATH_TEXT:
    raise ValueError("DIGER_SID_PATH is required")

REPORTS_DIR = Path(
    os.environ.get("SID_REPORTS_DIR", "reports/experiments")
)


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size="50m",
        history_days=None,
        protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
    ),
    catalog_from_fit=False,
    reports_dir=REPORTS_DIR,
    sid_pipeline=PrecomputedSidPipeline(
        PrecomputedSidConfig(
            path=Path(SID_PATH_TEXT),
            codebook_sizes=(256, 256, 256),
            item_id_space="original",
            expected_catalog_size=33_148,
        )
    ),
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=16,
            batch_size=1024,
            batch_size_val=128,
            num_workers=4,
            num_beams=150,
            recall_k=100,
            constrain_to_catalog=True,
            resolve_collisions=False,
            max_train_examples=None,
            max_validation_examples=None,
            max_test_examples=None,
            initialize_item_embeddings_from_content=False,
            use_bf16=False,
            auxiliary_item_loss_weight=0.0,
        )
    ),
    tracking=MLflowConfig(
        experiment_name="diger-common-generation-v0.1",
        run_name=f"diger-v0-1-{VARIANT}-poster-generation",
        tags=(
            ("method", "diger-reference-v0.1"),
            ("variant", VARIANT),
            ("protocol_id", POSTER_FULL_33148_PROTOCOL_ID),
            ("data_scope", "full"),
            ("evaluation_split", "legacy_heldout_reused"),
            ("downstream_recipe", "legacy_global_item_recall_selected"),
            ("collision_resolution", "disabled"),
            ("expected_catalog_size", "33148"),
            ("reference_commit", "cd72dcdc25d28e0acfa5aea14788ad863b41ab4f"),
        ),
    ),
)
