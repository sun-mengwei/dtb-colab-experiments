import numpy as np
import torch

from game_dtb.oscillatory_game import (
    OscillatoryGameParams,
    oscillatory_potential,
    oscillatory_vector_field,
    oscillatory_vector_field_jacobian,
    sample_uniform_initial_particles,
)


def test_analytic_pseudogradient_matches_potential_autograd() -> None:
    params = OscillatoryGameParams(
        lambda_=0.5, epsilon=0.5, omega=8.0 * np.pi, gamma=0.2
    )
    point = torch.tensor([0.31, -0.27], dtype=torch.float64)
    automatic = torch.func.jacrev(
        lambda value: oscillatory_potential(value, params)
    )(point)
    analytic = oscillatory_vector_field(point, params)
    assert torch.allclose(analytic, automatic, atol=1e-12, rtol=1e-12)


def test_analytic_game_jacobian_matches_autograd() -> None:
    params = OscillatoryGameParams(
        lambda_=0.5, epsilon=0.5, omega=16.0 * np.pi, gamma=0.2
    )
    point = torch.tensor([-0.19, 0.42], dtype=torch.float64)
    automatic = torch.func.jacrev(
        lambda value: oscillatory_vector_field(value, params)
    )(point)
    analytic = oscillatory_vector_field_jacobian(point, params)
    assert torch.allclose(analytic, automatic, atol=1e-12, rtol=1e-12)


def test_shared_initial_particles_are_reproducible() -> None:
    first = sample_uniform_initial_particles(2000, seed=2026)
    second = sample_uniform_initial_particles(2000, seed=2026)
    different = sample_uniform_initial_particles(2000, seed=2027)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (2000, 2)
    assert np.all((-1.0 <= first) & (first <= 1.0))
