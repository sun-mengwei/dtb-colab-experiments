from types import SimpleNamespace

import torch

from experiments.oscillatory_potential_game.experiment import make_model, model_state_hash
from game_dtb.algorithm import DTBConfig, NeuralDTBGameDynamics
from game_dtb.games import (
    OscillatoryGameParams,
    oscillatory_game_jacobian,
    oscillatory_game_velocity,
    oscillatory_potential,
)
from game_dtb.runner import run_dtb_trajectory
from game_dtb.state import uniform_box_particle_state


def test_analytic_velocity_matches_potential_autograd() -> None:
    params = OscillatoryGameParams(
        lambda_=0.5, epsilon=0.5, omega=8.0 * torch.pi, gamma=0.2
    )
    points = torch.tensor(
        [[0.31, -0.27], [-0.12, 0.43]], dtype=torch.float64
    )
    automatic = torch.vmap(torch.func.jacrev(lambda x: oscillatory_potential(x, params)))(
        points
    )
    analytic = oscillatory_game_velocity(points, params)
    assert torch.allclose(analytic, automatic, atol=1e-12, rtol=1e-12)


def test_analytic_game_jacobian_matches_autograd() -> None:
    params = OscillatoryGameParams(
        lambda_=0.5, epsilon=0.5, omega=16.0 * torch.pi, gamma=0.2
    )
    point = torch.tensor([-0.19, 0.42], dtype=torch.float64)
    automatic = torch.func.jacrev(
        lambda x: oscillatory_game_velocity(x, params)
    )(point)
    analytic = oscillatory_game_jacobian(point, params)
    assert torch.allclose(analytic, automatic, atol=1e-12, rtol=1e-12)


def test_uniform_initial_particles_are_reproducible() -> None:
    def sample(seed: int) -> torch.Tensor:
        return uniform_box_particle_state(
            200,
            2,
            -1.0,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(seed),
        ).particles

    first = sample(2026)
    second = sample(2026)
    different = sample(2027)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert bool(((first >= -1.0) & (first < 1.0)).all())


def test_model_initialization_is_controlled_by_model_seed() -> None:
    args = SimpleNamespace(
        architecture="mlp",
        width=6,
        depth=1,
        rank=3,
        activation="tanh",
        node_inner_steps=2,
        node_integration_time=1.0,
    )

    def initialized_hash(seed: int) -> str:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return model_state_hash(make_model(args, torch.float64))

    assert initialized_hash(91) == initialized_hash(91)
    assert initialized_hash(91) != initialized_hash(92)


def test_oscillatory_game_runs_through_shared_dtb_runner() -> None:
    dtype = torch.float64
    state = uniform_box_particle_state(
        8,
        2,
        -1.0,
        1.0,
        device=torch.device("cpu"),
        dtype=dtype,
        generator=torch.Generator().manual_seed(4),
    )
    torch.manual_seed(5)
    model = make_model(
        SimpleNamespace(
            architecture="mlp", width=4, depth=1, rank=2,
            activation="tanh", node_inner_steps=2, node_integration_time=1.0,
        ),
        dtype,
    )
    params = OscillatoryGameParams(omega=4.0 * torch.pi, gamma=0.2)
    method = NeuralDTBGameDynamics(
        model,
        drift=lambda x: oscillatory_game_velocity(x, params),
        diffusion=torch.zeros(2, 2, dtype=dtype),
        config=DTBConfig(
            step_size=0.005,
            basis_size=6,
            svd_rtol=1e-5,
            jacobian_chunk_size=8,
            derivative_chunk_size=8,
            seed=6,
        ),
    )
    result = run_dtb_trajectory(
        method,
        state,
        steps=2,
        potential=lambda x: oscillatory_potential(x, params),
    )

    assert result.particles.shape == (3, 8, 2)
    assert result.projection_residuals.shape == (2,)
    assert result.retained_rank.shape == (2,)
    assert result.mean_potential is not None
    assert result.mean_potential.shape == (3,)
    assert torch.isfinite(torch.from_numpy(result.particles)).all()
    assert not torch.equal(result.final_state.particles, state.particles)
