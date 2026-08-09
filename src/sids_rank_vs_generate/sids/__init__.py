"""SID-specific code and concrete pipelines for different types of SIDs.

Some SIDs will require a kind-of-hardcoded training pipeline (for example, RQ-KMeans is
a bit different from regular training).
"""

from .base import ISidPipeline
from .config import PlumRQVAEConfig, RQKMeansConfig, RQVAEConfig
from .plum_rq_vae_pipeline import PlumRQVAEPipeline
from .r3vae import R3VAEConfig, R3VAESid
from .rq_kmeans_pipeline import RQKMeansPipeline
from .rq_vae_pipeline import RQVAEPipeline

__all__ = [
    "ISidPipeline",
    "PlumRQVAEPipeline",
    "PlumRQVAEConfig",
    "RQKMeansConfig",
    "RQKMeansPipeline",
    "RQVAEConfig",
    "RQVAEPipeline",
    "R3VAEConfig",
    "R3VAESid",
]
