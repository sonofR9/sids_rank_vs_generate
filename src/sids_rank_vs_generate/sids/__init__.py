"""SID-specific code and concrete pipelines for different types of SIDs.

Some SIDs will require a kind-of-hardcoded training pipeline (for example, RQ-KMeans is
a bit different from regular training).
"""

from .base import ISidPipeline
from .config import (
    PlumRQVAEConfig,
    RQKMeansConfig,
    RQVAEConfig,
    UserAwareRQKMeansConfig,
)
from .plum_rq_vae_pipeline import PlumRQVAEPipeline
from .rq_kmeans_pipeline import RQKMeansPipeline
from .rq_vae_pipeline import RQVAEPipeline
from .user_aware import (
    IUserAwareSidPipeline,
    UserAwareSidTrainingData,
    user_sid_columns,
    validate_user_sid_output,
)
from .user_aware_rq_kmeans import UserAwareRQKMeansPipeline
from .transformer_user_sid_mapper import (
    TransformerUserSidMapper,
    UserSidMapperConfig,
)
from .user_sid_mapper import (
    FrequencyUserSidMapper,
    IUserSidMapper,
    mapping_ambiguity_metrics,
    mapping_fit_support_metrics,
)

__all__ = [
    "ISidPipeline",
    "IUserAwareSidPipeline",
    "IUserSidMapper",
    "FrequencyUserSidMapper",
    "PlumRQVAEPipeline",
    "PlumRQVAEConfig",
    "RQKMeansConfig",
    "RQKMeansPipeline",
    "RQVAEConfig",
    "RQVAEPipeline",
    "UserAwareSidTrainingData",
    "UserAwareRQKMeansConfig",
    "UserAwareRQKMeansPipeline",
    "TransformerUserSidMapper",
    "UserSidMapperConfig",
    "mapping_ambiguity_metrics",
    "mapping_fit_support_metrics",
    "user_sid_columns",
    "validate_user_sid_output",
]
