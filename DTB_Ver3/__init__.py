"""Small experiment package for deterministic and stochastic DTB games."""

from .experiment import (
    DTBExperiment,
    ExperimentConfig,
    ExperimentResult,
    StepSizeSweepResult,
    run_experiment,
    run_step_size_sweep,
    rk4_flow,
    rk4_step,
)
from .diagnostics import projection_metrics, run_oscillatory_dynamic_diagnostic
from .games import (
    BlockCournotGame,
    ConstantDiffusion,
    CournotGame,
    Diffusion,
    FunctionalDiffusion,
    FunctionalGame,
    Game,
    OscillatoryGame,
    OscillatoryNonpotentialGame,
)
from .models import MMNN, MMNNLayer, ResidualMLP, ResidualMMNN, TangentMLP

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
    "OscillatoryNonpotentialGame",
    "MMNN",
    "MMNNLayer",
    "ResidualMLP",
    "ResidualMMNN",
    "StepSizeSweepResult",
    "TangentMLP",
    "run_experiment",
    "run_step_size_sweep",
    "rk4_flow",
    "rk4_step",
    "projection_metrics",
    "run_oscillatory_dynamic_diagnostic",
]
