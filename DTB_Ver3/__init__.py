"""Small experiment package for deterministic and stochastic DTB games."""

from .experiment import (
    DTBExperiment,
    ExperimentConfig,
    ExperimentResult,
    StepSizeSweepResult,
    run_experiment,
    run_step_size_sweep,
)
from .games import (
    BlockCournotGame,
    ConstantDiffusion,
    CournotGame,
    Diffusion,
    FunctionalDiffusion,
    FunctionalGame,
    Game,
    OscillatoryGame,
)
from .models import ResidualMLP, TangentMLP

__all__ = [
    "BlockCournotGame",
    "ConstantDiffusion",
    "CournotGame",
    "DTBExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "Diffusion",
    "FunctionalDiffusion",
    "FunctionalGame",
    "Game",
    "OscillatoryGame",
    "ResidualMLP",
    "StepSizeSweepResult",
    "TangentMLP",
    "run_experiment",
    "run_step_size_sweep",
]
