import math
from types import SimpleNamespace

import torch

from game_dtb.games import (
    cournot_duopoly_drift,
    cournot_five_player_drift,
    cournot_multiplayer_payoff,
    cournot_three_player_drift,
    nonlinear_network_drift,
    nonlinear_network_jacobian,
    nonlinear_network_payoff,
)
from game_dtb.network_analysis import refine_network_equilibrium
from game_dtb.runner import (
    _equilibria,
    _equilibrium_stability,
    diffusion_entry_from_noise,
    nonlinear_network_parameters,
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


def test_five_player_constrained_cournot_equilibria() -> None:
    points = [
        torch.zeros(5, dtype=torch.float64),
        torch.full((5,), 7.0 / 32.0, dtype=torch.float64),
    ]
    for zero_index in range(5):
        point = torch.full((5,), 5.0 / 18.0, dtype=torch.float64)
        point[zero_index] = 0.0
        points.append(point)
    equilibria = torch.stack(points)
    drift = cournot_five_player_drift(equilibria, b=1.0, mu=2.0)
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-12)


def test_five_player_equilibria_and_stability_labels() -> None:
    args = SimpleNamespace(game="cournot5", cournot_b=1.0, cournot_mu=2.0)
    equilibria = _equilibria(args)
    labels = _equilibrium_stability(args)
    assert equilibria.shape == (7, 5)
    assert labels.tolist() == [False] * 7


def test_each_five_player_equilibrium_has_unstable_direction() -> None:
    points = [torch.zeros(5), torch.full((5,), 7.0 / 32.0)]
    for zero_index in range(5):
        point = torch.full((5,), 5.0 / 18.0)
        point[zero_index] = 0.0
        points.append(point)
    for point in points:
        jacobian = torch.func.jacrev(cournot_five_player_drift)(point.to(torch.float64))
        assert bool((torch.linalg.eigvals(jacobian).real > 0).any())


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
    assert snapshot_schedule("auto", 2, 0.01) == {0: 0.0, 1: 0.01, 2: 0.02}


def test_nonlinear_network_payoff_gradient_matches_drift() -> None:
    x = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)
    matrix = torch.tensor(
        [[0.0, 0.4, -0.2], [-0.1, 0.0, 0.5], [0.3, -0.6, 0.0]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    mu = torch.tensor([0.9, 1.1, 0.8], dtype=torch.float64)
    beta = torch.tensor([0.7, 0.6, 0.9], dtype=torch.float64)
    payoff_jacobian = torch.func.jacrev(
        lambda value: nonlinear_network_payoff(value, matrix, bias, mu, beta)
    )(x)
    assert torch.allclose(
        payoff_jacobian.diagonal(),
        nonlinear_network_drift(x, matrix, bias, mu, beta),
        atol=1e-12,
    )


def test_nonlinear_network_analytic_jacobian_matches_autodiff() -> None:
    x = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)
    matrix = torch.tensor(
        [[0.0, 0.4, -0.2], [-0.1, 0.0, 0.5], [0.3, -0.6, 0.0]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    mu = torch.tensor([0.9, 1.1, 0.8], dtype=torch.float64)
    beta = torch.tensor([0.7, 0.6, 0.9], dtype=torch.float64)
    automatic = torch.func.jacrev(
        lambda value: nonlinear_network_drift(value, matrix, bias, mu, beta)
    )(x)
    analytic = nonlinear_network_jacobian(x, matrix, mu, beta)
    assert torch.allclose(analytic, automatic, atol=1e-12)


def test_network_generator_is_reproducible_and_normalized() -> None:
    args = SimpleNamespace(
        dim=12,
        network_density=0.4,
        network_scale=0.75,
        network_seed=17,
        network_bias_std=0.15,
        network_mu=1.0,
        network_beta=0.8,
    )
    first = nonlinear_network_parameters(args)
    second = nonlinear_network_parameters(args)
    assert all(
        torch.equal(torch.from_numpy(a), torch.from_numpy(b))
        for a, b in zip(first, second)
    )
    matrix = first[0]
    assert torch.count_nonzero(torch.from_numpy(matrix).diagonal()) == 0
    spectral_radius = max(abs(torch.linalg.eigvals(torch.from_numpy(matrix))))
    assert math.isclose(float(spectral_radius), 0.75, rel_tol=1e-10)


def test_network_equilibrium_refinement_finds_decoupled_stable_root() -> None:
    matrix = torch.zeros((2, 2), dtype=torch.float64).numpy()
    bias = torch.zeros(2, dtype=torch.float64).numpy()
    mu = torch.ones(2, dtype=torch.float64).numpy()
    beta = torch.zeros(2, dtype=torch.float64).numpy()
    root, residual = refine_network_equilibrium(
        torch.tensor([0.8, -0.8], dtype=torch.float64).numpy(),
        matrix,
        bias,
        mu,
        beta,
    )
    assert residual < 1e-7
    assert torch.allclose(
        torch.from_numpy(root), torch.tensor([1.0, -1.0], dtype=torch.float64),
        atol=1e-6,
    )
