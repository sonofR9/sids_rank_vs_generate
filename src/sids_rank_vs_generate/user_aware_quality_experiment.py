from __future__ import annotations

from pathlib import Path

from .data.protocols import POSTER_FULL_33148_PROTOCOL_ID
from .evaluation.generation_stage import GenerationConfig
from .evaluation.sid_quality import UserCatalogQualityConfig
from .experiment import ExperimentConfig, TrainingConfig, YambdaDatasetConfig
from .sids.config import RQKMeansConfig, UserAwareRQKMeansConfig
from .sids.level_permutation import LevelPermutationUserAwareSidPipeline
from .sids.ranker_embedding_user_aware import (
    RankerEmbeddingUserAwareRQKMeansConfig,
    RankerEmbeddingUserAwareRQKMeansPipeline,
)
from .sids.rq_kmeans_pipeline import RQKMeansPipeline
from .sids.transformer_user_sid_mapper import UserSidMapperConfig
from .sids.user_aware import IUserAwareSidPipeline
from .sids.user_aware_rq_kmeans import UserAwareRQKMeansPipeline
from .tracking import MLflowConfig
from .user_aware_quality_matrix import (
    EXPERIMENT_NAME,
    GLOBAL_SID_CODEBOOK_SIZES,
    USER_AWARE_QUALITY_MATRIX,
    V1_FAMILY,
    V2_FAMILY,
    UserAwareQualityRun,
)


FIXED_MAPPER_CONFIG = UserSidMapperConfig(
    kind="transformer",
    d_model=192,
    d_ff=512,
    num_heads=6,
    num_encoder_layers=3,
    num_decoder_layers=3,
    batch_size=512,
    inference_batch_size=512,
    lr=3e-4,
    max_epochs=30,
    early_stopping_patience=4,
    early_stopping_min_delta=0.0,
    catalog_masked_teacher_forced_loss=True,
    collision_level_loss_weight=None,
    validation_selection_metric="autoregressive_catalog_top_k_accuracy",
    validation_selection_top_k=3,
    decode_beam_size=16,
)


FIXED_GENERATION_CONFIG = GenerationConfig(
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
    mapper_top_k=3,
    mapper=FIXED_MAPPER_CONFIG,
    run_diagnostics=False,
    use_user_profile=True,
    initialize_item_embeddings_from_content=False,
    sid_only_validation_during_training=True,
    validation_metric="sid_beam_recall@100",
    use_bf16=False,
    auxiliary_item_loss_weight=0.0,
)

FIXED_DATASET_CONFIG = YambdaDatasetConfig(
    size="50m",
    history_days=None,
    protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
)

FIXED_GLOBAL_SID_CONFIG = RQKMeansConfig(
    codebook_sizes=GLOBAL_SID_CODEBOOK_SIZES,
)

FIXED_USER_CATALOG_QUALITY_CONFIG = UserCatalogQualityConfig(
    enabled=True,
    max_users=256,
    selection="stable_hash",
    seed=42,
)


def get_user_aware_quality_run(run_name: str) -> UserAwareQualityRun:
    """Найти строку согласованной матрицы"""

    matches = [
        run for run in USER_AWARE_QUALITY_MATRIX if run.run_name == run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown matrix run: {run_name}")
    return matches[0]


def _base_user_sid_pipeline(
    run: UserAwareQualityRun,
) -> IUserAwareSidPipeline:
    if run.family == V1_FAMILY:
        return UserAwareRQKMeansPipeline(
            UserAwareRQKMeansConfig(
                codebook_sizes=run.personal_codebook_sizes,
                user_profile_weight=run.user_profile_weight,
            )
        )
    if run.family == V2_FAMILY:
        assert run.fusion_weight is not None
        return RankerEmbeddingUserAwareRQKMeansPipeline(
            RankerEmbeddingUserAwareRQKMeansConfig(
                codebook_sizes=run.personal_codebook_sizes,
                user_profile_weight=run.user_profile_weight,
                ranker_hidden_dims=(256, 128),
                ranker_representation_dim=64,
                ranker_dropout=0.1,
                fusion_weight=run.fusion_weight,
                ranker_label_fraction=0.2,
                negative_ratio=10,
            )
        )
    raise ValueError(f"unsupported user-aware family: {run.family}")


def build_user_sid_pipeline(
    run: UserAwareQualityRun,
) -> IUserAwareSidPipeline:
    """Построить SID pipeline без обучения"""

    base = _base_user_sid_pipeline(run)
    identity = tuple(range(len(run.personal_codebook_sizes)))
    if run.level_permutation == identity:
        return base
    return LevelPermutationUserAwareSidPipeline(
        base,
        codebook_sizes=run.personal_codebook_sizes,
        level_permutation=run.level_permutation,
    )


def _validate_user_sid_pipeline(
    actual: IUserAwareSidPipeline | None,
    expected: IUserAwareSidPipeline,
) -> None:
    if actual is None or type(actual) is not type(expected):
        raise ValueError("user SID pipeline differs from the matrix row")
    if getattr(actual, "config", None) != getattr(expected, "config", None):
        raise ValueError("user SID config differs from the matrix row")
    if isinstance(expected, LevelPermutationUserAwareSidPipeline):
        assert isinstance(actual, LevelPermutationUserAwareSidPipeline)
        _validate_user_sid_pipeline(actual.pipeline, expected.pipeline)


def validate_user_aware_quality_experiment(
    config: ExperimentConfig,
    run: UserAwareQualityRun,
) -> None:
    """Проверить фактический конфиг перед запуском"""

    known = get_user_aware_quality_run(run.run_name)
    if run != known:
        raise ValueError("run does not match the declared matrix row")
    if config.dataset != FIXED_DATASET_CONFIG:
        raise ValueError("dataset config differs from the fixed protocol")
    if config.catalog_from_fit is not False:
        raise ValueError("matrix requires the full-data catalog")
    if type(config.sid_pipeline) is not RQKMeansPipeline:
        raise ValueError("global SID pipeline differs from the fixed baseline")
    if config.sid_pipeline.config != FIXED_GLOBAL_SID_CONFIG:
        raise ValueError("global SID config differs from the fixed baseline")
    _validate_user_sid_pipeline(
        config.user_sid_pipeline,
        build_user_sid_pipeline(known),
    )
    if config.user_catalog_quality != FIXED_USER_CATALOG_QUALITY_CONFIG:
        raise ValueError("user catalog quality config differs from the matrix")
    if config.training.generation != FIXED_GENERATION_CONFIG:
        raise ValueError("generation config differs from the fixed recipe")
    if config.tracking.experiment_name != EXPERIMENT_NAME:
        raise ValueError("MLflow experiment name differs from the matrix")
    if config.tracking.run_name != known.run_name:
        raise ValueError("MLflow run name differs from the matrix row")
    if config.tracking.tags != tuple(known.tags.items()):
        raise ValueError("MLflow tags differ from the matrix row")


def build_user_aware_quality_experiment(
    run: UserAwareQualityRun,
    *,
    reports_dir: Path = Path("reports/experiments"),
    tracking_enabled: bool = False,
    tracking_uri: str | None = None,
) -> ExperimentConfig:
    """Построить общий ExperimentConfig"""

    known = get_user_aware_quality_run(run.run_name)
    if run != known:
        raise ValueError("run does not match the declared matrix row")
    config = ExperimentConfig(
        dataset=FIXED_DATASET_CONFIG,
        catalog_from_fit=False,
        reports_dir=Path(reports_dir),
        sid_pipeline=RQKMeansPipeline(
            FIXED_GLOBAL_SID_CONFIG
        ),
        user_sid_pipeline=build_user_sid_pipeline(run),
        user_catalog_quality=FIXED_USER_CATALOG_QUALITY_CONFIG,
        training=TrainingConfig(generation=FIXED_GENERATION_CONFIG),
        tracking=MLflowConfig(
            enabled=tracking_enabled,
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT_NAME,
            run_name=run.run_name,
            tags=tuple(run.tags.items()),
        ),
    )
    validate_user_aware_quality_experiment(config, known)
    return config


def build_user_aware_quality_experiment_by_name(
    run_name: str,
    **kwargs,
) -> ExperimentConfig:
    """Построить конфиг по имени строки"""

    return build_user_aware_quality_experiment(
        get_user_aware_quality_run(run_name),
        **kwargs,
    )
