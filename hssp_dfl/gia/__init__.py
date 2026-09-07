"""Gradient-inversion adapters and evaluation helpers."""

from .breaching_adapter import (
    BREACHING_COMMIT,
    BREACHING_REPOSITORY,
    SUPPORTED_ATTACKS,
    AttackResult,
    DataSpec,
    gradients_from_batch,
    gradients_from_model_delta,
    run_breaching_attack,
)
from .metrics import align_and_score_batch

__all__ = [
    "BREACHING_COMMIT",
    "BREACHING_REPOSITORY",
    "SUPPORTED_ATTACKS",
    "AttackResult",
    "DataSpec",
    "align_and_score_batch",
    "gradients_from_batch",
    "gradients_from_model_delta",
    "run_breaching_attack",
]
