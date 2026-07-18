"""SID-specific code and concrete pipelines for different types of SIDs.

Some SIDs will require a kind-of-hardcoded training pipeline (for example, RQ-KMeans is
a bit different from regular training).
"""

from .base import ISidPipeline

__all__ = ["ISidPipeline"]
