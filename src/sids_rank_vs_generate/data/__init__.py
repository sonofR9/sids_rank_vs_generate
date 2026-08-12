"""Shared interaction-data loading and preprocessing."""

from .preprocessing import (
    PreparedInteractions,
    PreprocessingConfig,
    build_histories,
    build_targets,
    interaction_fingerprint,
    load_and_prepare_interactions,
    load_interactions,
    prepare_interactions,
)
from .protocols import (
    POSTER_FULL_33148_CATALOG_SIZE,
    POSTER_FULL_33148_DATA_CONTRACT,
    POSTER_FULL_33148_PROTOCOL_ID,
    POSTER_RQKMEANS_L3_C256_REFERENCE,
    POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE,
    THREE_WAY_PROTOCOL_ID,
    expected_catalog_size,
    validate_poster_full_33148_summary,
)
from .user_aware_behavior import (
    UserAwareBehaviorData,
    UserAwareBehaviorSummary,
    load_fit_user_aware_behavior,
    user_aware_behavior_fingerprint,
)

__all__ = [
    "PreparedInteractions",
    "PreprocessingConfig",
    "build_histories",
    "build_targets",
    "interaction_fingerprint",
    "load_and_prepare_interactions",
    "load_interactions",
    "prepare_interactions",
    "POSTER_FULL_33148_CATALOG_SIZE",
    "POSTER_FULL_33148_DATA_CONTRACT",
    "POSTER_FULL_33148_PROTOCOL_ID",
    "POSTER_RQKMEANS_L3_C256_REFERENCE",
    "POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE",
    "THREE_WAY_PROTOCOL_ID",
    "expected_catalog_size",
    "validate_poster_full_33148_summary",
    "UserAwareBehaviorData",
    "UserAwareBehaviorSummary",
    "load_fit_user_aware_behavior",
    "user_aware_behavior_fingerprint",
]
