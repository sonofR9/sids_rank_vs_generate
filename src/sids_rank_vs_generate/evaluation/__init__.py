"""Evaluation interfaces and ranking metrics shared by the project."""

from .metrics import evaluate_ranking, get_ranking_metrics
from .generation_stage import GenerationConfig, GenerationStage
from .ranking_stage import RankingConfig, RankingStage, RankingTrainingSchedule
from .sid_quality import (
    SID_QUALITY_SCHEMA_VERSION,
    UserCatalogQualityConfig,
    calculate_sid_quality,
    calculate_user_catalog_sid_quality,
    calculate_user_catalog_sid_quality_from_frame,
    select_stable_quality_users,
)
from .stages import (
    DEFAULT_STAGE_FACTORIES,
    EvaluationContext,
    IEvaluationStage,
    create_evaluation_stage,
)

__all__ = [
    "DEFAULT_STAGE_FACTORIES",
    "EvaluationContext",
    "GenerationConfig",
    "GenerationStage",
    "IEvaluationStage",
    "RankingConfig",
    "RankingStage",
    "RankingTrainingSchedule",
    "SID_QUALITY_SCHEMA_VERSION",
    "UserCatalogQualityConfig",
    "calculate_sid_quality",
    "calculate_user_catalog_sid_quality",
    "calculate_user_catalog_sid_quality_from_frame",
    "create_evaluation_stage",
    "evaluate_ranking",
    "get_ranking_metrics",
    "select_stable_quality_users",
]
