"""Small experiment package for deterministic Deep Tangent Bundle games."""

from .experiment import (
    DTBExperiment,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)
from .games import BlockCournotGame, CournotGame, FunctionalGame, Game

__all__ = [
    "BlockCournotGame",
    "CournotGame",
    "DTBExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "FunctionalGame",
    "Game",
    "run_experiment",
]
