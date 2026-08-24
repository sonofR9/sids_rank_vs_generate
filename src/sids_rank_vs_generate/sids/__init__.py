"""SID-specific code and concrete pipelines for different types of SIDs.

Some SIDs will require a kind-of-hardcoded training pipeline (for example, RQ-KMeans is
a bit different from regular training).
"""

from .base import ISidPipeline, SidTrainingData
from .config import (
    CollaborativeAlignmentConfig,
    CollaborativeAwareRQVAEConfig,
    CollaborativeAwareRQKMeansConfig,
    PlumRQVAEConfig,
    RQKMeansConfig,
    RQVAEConfig,
    RQVAELetterConfig,
    UserAwareRQKMeansConfig,
)
from .collaborative_rq_kmeans import (
    CollaborativeAwareRQKMeansPipeline,
    mine_collaborative_pairs,
)
from .collaborative_rq_vae import CollaborativeAwareRQVAEPipeline
from .plum_rq_vae_pipeline import PlumRQVAEPipeline
from .rq_kmeans_pipeline import RQKMeansPipeline
from .rq_vae_pipeline import RQVAEPipeline
from .bpe_rq_vae_pipeline import BPE_RQVAEConfig, BPERQVAEPipeline
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
from .rqvae_letter_pipeline import RQVAELetterPipeline
__all__ = [
    "ISidPipeline",
    "BPE_RQVAEConfig",
    "BPERQVAEPipeline",
    "RQVAEPipeline",
    "RQVAELetterConfig",
    "RQVAELetterPipeline",
    "SidTrainingData",
    "CollaborativeAlignmentConfig",
    "CollaborativeAwareRQVAEConfig",
    "CollaborativeAwareRQVAEPipeline",
    "CollaborativeAwareRQKMeansConfig",
    "CollaborativeAwareRQKMeansPipeline",
    "mine_collaborative_pairs",
    "IUserAwareSidPipeline",
    "IUserSidMapper",
    "FrequencyUserSidMapper",
    "PlumRQVAEPipeline",
    "PlumRQVAEConfig",
    "RQKMeansConfig",
    "RQKMeansPipeline",
    "RQVAEConfig",
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
