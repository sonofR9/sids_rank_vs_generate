"""Масштабирование генератора для максимального Recall@100"""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.data import (
    POSTER_FULL_33148_PROTOCOL_ID,
    THREE_WAY_PROTOCOL_ID,
)
from sids_rank_vs_generate.evaluation import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import (
    PrecomputedSidConfig,
    PrecomputedSidPipeline,
    RQKMeansConfig,
    RQKMeansPipeline,
)
from sids_rank_vs_generate.tracking import MLflowConfig


METHOD = os.environ.get("DIGER_SID_METHOD", "diger").strip().lower()
if METHOD not in {"diger", "rqkmeans"}:
    raise ValueError("DIGER_SID_METHOD must be diger or rqkmeans")

VARIANT = os.environ.get("DIGER_VARIANT", "both").strip().lower()
if METHOD == "diger" and VARIANT not in {"both", "frqud", "sdud"}:
    raise ValueError("DIGER_VARIANT must be both, frqud or sdud")
if METHOD == "rqkmeans":
    VARIANT = "rqkmeans"

MODEL_VARIANT = os.environ.get("DIGER_RECALL_MODEL", "strong128").strip().lower()
if MODEL_VARIANT not in {"strong128", "large256"}:
    raise ValueError("DIGER_RECALL_MODEL must be strong128 or large256")

PROTOCOL = os.environ.get("DIGER_RECALL_PROTOCOL", "poster").strip().lower()
if PROTOCOL == "poster":
    protocol_id = POSTER_FULL_33148_PROTOCOL_ID
    max_validation_examples = None
elif PROTOCOL == "three-way":
    protocol_id = THREE_WAY_PROTOCOL_ID
    max_validation_examples = 10_000
else:
    raise ValueError("DIGER_RECALL_PROTOCOL must be poster or three-way")

if MODEL_VARIANT == "strong128":
    model = {
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 4,
        "num_layers": 6,
        "batch_size": 1024,
        "lr": 1e-3,
        "pad_content_embeddings_to_model_dim": False,
    }
else:
    model = {
        "d_model": 256,
        "d_ff": 1024,
        "num_heads": 8,
        "num_layers": 8,
        "batch_size": 512,
        "lr": 5e-4,
        "pad_content_embeddings_to_model_dim": True,
    }

if METHOD == "diger":
    sid_path_text = os.environ.get("DIGER_SID_PATH")
    if not sid_path_text:
        raise ValueError("DIGER_SID_PATH is required for DIGER")
    sid_pipeline = PrecomputedSidPipeline(
        PrecomputedSidConfig(
            path=Path(sid_path_text),
            codebook_sizes=(256, 256, 256),
            item_id_space="original",
            expected_catalog_size=33_148,
        )
    )
else:
    sid_pipeline = RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256)
    )

reports_dir = Path(
    os.environ.get("SID_REPORTS_DIR", "reports/experiments")
)
run_name = f"{VARIANT}-{MODEL_VARIANT}-{PROTOCOL}-recall-scale"


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size="50m",
        history_days=None,
        protocol_id=protocol_id,
    ),
    catalog_from_fit=False,
    reports_dir=reports_dir,
    sid_pipeline=sid_pipeline,
    training=TrainingConfig(
        generation=GenerationConfig(
            max_history_len=64,
            d_model=model["d_model"],
            d_ff=model["d_ff"],
            num_heads=model["num_heads"],
            num_layers=model["num_layers"],
            batch_size=model["batch_size"],
            batch_size_val=128,
            num_workers=8,
            lr=model["lr"],
            max_epochs=40,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
            num_beams=150,
            recall_k=100,
            constrain_to_catalog=True,
            resolve_collisions=False,
            max_train_examples=None,
            max_validation_examples=max_validation_examples,
            max_test_examples=None,
            initialize_item_embeddings_from_content=True,
            content_embedding_scale=model["d_model"] ** 0.5,
            pad_content_embeddings_to_model_dim=model[
                "pad_content_embeddings_to_model_dim"
            ],
            auxiliary_item_loss_weight=0.1,
            ranking_temperature=0.07,
            logq_mode="corrected",
            logq_debug=True,
            gather_distributed_negatives=False,
            use_bf16=False,
        )
    ),
    tracking=MLflowConfig(
        experiment_name="diger-recall-scale-v0.1",
        run_name=run_name,
        tags=(
            ("series", "diger-recall-scale-v0.1"),
            ("sid_method", METHOD),
            ("variant", VARIANT),
            ("model_variant", MODEL_VARIANT),
            ("protocol", PROTOCOL),
            ("protocol_id", protocol_id),
            ("objective", "maximize-recall-at-100"),
            ("history_length", "64"),
            ("content_initialization", "true"),
            ("auxiliary_item_loss_weight", "0.1"),
            ("logq_mode", "corrected"),
            ("expected_catalog_size", "33148"),
        ),
    ),
)
