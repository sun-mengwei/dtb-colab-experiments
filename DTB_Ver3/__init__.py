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
from .dtb import evaluate_dtb_projection
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
from .refit import (
    NetworkRefitConfig,
    NetworkRefitDiagnostics,
    RefitDTBConfig,
    RefitDTBResult,
    refit_network_to_particles,
    run_refit_dtb,
)
from .representation import (
    RepresentationComparisonConfig,
    RepresentationComparisonResult,
    run_representation_comparison,
)

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
    "NetworkRefitConfig",
    "NetworkRefitDiagnostics",
    "RefitDTBConfig",
    "RefitDTBResult",
    "RepresentationComparisonConfig",
    "RepresentationComparisonResult",
    "MMNN",
    "MMNNLayer",
    "ResidualMLP",
    "ResidualMMNN",
    "StepSizeSweepResult",
    "TangentMLP",
    "run_experiment",
    "run_refit_dtb",
    "run_representation_comparison",
    "run_step_size_sweep",
    "rk4_flow",
    "rk4_step",
    "projection_metrics",
    "run_oscillatory_dynamic_diagnostic",
    "evaluate_dtb_projection",
    "refit_network_to_particles",
]
