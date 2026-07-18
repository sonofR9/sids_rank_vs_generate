"""Evaluation interfaces shared by the common runner."""

from .stages import (
    DEFAULT_STAGE_FACTORIES,
    EvaluationContext,
    IEvaluationStage,
    create_evaluation_stage,
)

__all__ = [
    "DEFAULT_STAGE_FACTORIES",
    "EvaluationContext",
    "IEvaluationStage",
    "create_evaluation_stage",
]
