import math
from types import SimpleNamespace

import torch

from game_dtb.games import (
    cournot_duopoly_drift,
    cournot_multiplayer_payoff,
    cournot_three_player_drift,
)
from game_dtb.runner import (
    _equilibrium_stability,
    diffusion_entry_from_noise,
    snapshot_schedule,
)


def test_target_cournot_equilibria() -> None:
    equilibria = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    drift = cournot_duopoly_drift(equilibria, b=1.0, mu=2.0)
    assert torch.allclose(drift, torch.zeros_like(drift))


def test_three_player_cournot_equilibria() -> None:
    equilibria = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )
    drift = cournot_three_player_drift(equilibria, b=1.0, mu=2.0)
    assert torch.allclose(drift, torch.zeros_like(drift))


def test_equilibrium_consistent_payoff_gradient_matches_drift() -> None:
    x = torch.tensor([0.17, 0.31, 0.22], dtype=torch.float64)
    payoff_jacobian = torch.func.jacrev(cournot_multiplayer_payoff)(x)
    own_action_gradient = payoff_jacobian.diagonal()
    assert torch.allclose(own_action_gradient, cournot_three_player_drift(x))


def test_printed_full_total_payoff_is_inconsistent_with_reported_equilibrium() -> None:
    x = torch.full((3,), 3.0 / 8.0, dtype=torch.float64)
    total = x.sum()
    printed_gradient = 4.0 * (
        total * (1.0 - total) + x * (1.0 - 2.0 * total)
    )
    assert not torch.allclose(printed_gradient, torch.zeros_like(x))


def test_three_player_equilibrium_stability_labels() -> None:
    args = SimpleNamespace(game="cournot3", cournot_b=1.0, cournot_mu=2.0)
    labels = _equilibrium_stability(args)
    assert labels.tolist() == [False, True, True, True, True]


def test_three_player_origin_has_unstable_direction() -> None:
    origin = torch.zeros(3, dtype=torch.float64)
    jacobian = torch.func.jacrev(lambda x: cournot_three_player_drift(x))(origin)
    assert bool((torch.linalg.eigvals(jacobian).real > 0).any())


def test_thesis_noise_amplitude_gives_algorithm_diffusion() -> None:
    assert math.isclose(diffusion_entry_from_noise(0.1), 0.01)


def test_figure_snapshot_schedule() -> None:
    schedule = snapshot_schedule("0,0.2,0.4,0.6,0.8,1.0", 50, 0.02)
    assert schedule == {0: 0.0, 10: 0.2, 20: 0.4, 30: 0.6, 40: 0.8, 50: 1.0}
