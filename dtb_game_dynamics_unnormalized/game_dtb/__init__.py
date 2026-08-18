"""Neural Deep Tangent Bundle dynamics for distributional games."""

from .algorithm import DTBConfig, NeuralDTBGameDynamics
from .games import (
    cournot_duopoly_drift,
    cournot_five_player_drift,
    cournot_multiplayer_payoff,
    cournot_multiplayer_drift,
    cournot_three_player_drift,
    linear_quadratic_drift,
)
from .models import TangentMLP, TangentMMNN
from .state import ParticleState, gaussian_particle_state, uniform_box_particle_state

__all__ = [
    "DTBConfig",
    "NeuralDTBGameDynamics",
    "ParticleState",
    "TangentMLP",
    "TangentMMNN",
    "cournot_duopoly_drift",
    "cournot_five_player_drift",
    "cournot_multiplayer_payoff",
    "cournot_multiplayer_drift",
    "cournot_three_player_drift",
    "gaussian_particle_state",
    "linear_quadratic_drift",
    "uniform_box_particle_state",
]
