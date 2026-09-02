import math

import torch

from game_dtb.state import gaussian_particle_state, uniform_box_particle_state


def test_standard_gaussian_initialization_formula() -> None:
    state = gaussian_particle_state(
        4,
        2,
        0.0,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(5),
    )
    expected_log_density = -0.5 * state.particles.square().sum(dim=1) - math.log(2.0 * math.pi)
    assert torch.allclose(state.log_density, expected_log_density)
    assert torch.allclose(state.score, -state.particles)
    assert torch.equal(state.labels, state.particles)


def test_unit_box_uniform_has_zero_interior_score() -> None:
    state = uniform_box_particle_state(
        100,
        2,
        0.0,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(9),
    )
    assert bool(((state.particles >= 0.0) & (state.particles < 1.0)).all())
    assert torch.equal(state.score, torch.zeros_like(state.score))
    assert torch.equal(state.log_density, torch.zeros_like(state.log_density))
