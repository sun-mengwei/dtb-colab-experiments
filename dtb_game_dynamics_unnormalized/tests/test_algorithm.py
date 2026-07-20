import torch

from game_dtb.algorithm import DTBConfig, NeuralDTBGameDynamics
from game_dtb.games import cournot_three_player_drift
from game_dtb.models import TangentMLP, TangentMMNN
from game_dtb.state import gaussian_particle_state


def test_zero_target_keeps_state_fixed() -> None:
    torch.manual_seed(1)
    dtype = torch.float64
    device = torch.device("cpu")
    state = gaussian_particle_state(
        5, 2, 0.0, 1.0, device=device, dtype=dtype,
        generator=torch.Generator().manual_seed(2),
    )
    model = TangentMLP(2, width=4, depth=1, dtype=dtype)
    config = DTBConfig(
        step_size=0.01,
        basis_size=6,
        svd_rtol=1e-10,
        jacobian_chunk_size=5,
        derivative_chunk_size=5,
        seed=3,
    )
    method = NeuralDTBGameDynamics(
        model,
        drift=lambda x: torch.zeros_like(x),
        diffusion=torch.zeros(2, 2, dtype=dtype),
        config=config,
    )
    result = method.step(state, 0)

    assert torch.allclose(result.state.particles, state.particles)
    assert torch.allclose(result.state.log_density, state.log_density)
    assert torch.allclose(result.state.score, state.score)
    assert result.alpha_norm == 0.0


def test_three_dimensional_step_is_finite() -> None:
    torch.manual_seed(4)
    dtype = torch.float64
    state = gaussian_particle_state(
        4,
        3,
        0.5,
        0.15,
        device=torch.device("cpu"),
        dtype=dtype,
        generator=torch.Generator().manual_seed(5),
    )
    model = TangentMLP(3, width=4, depth=1, dtype=dtype)
    method = NeuralDTBGameDynamics(
        model,
        drift=cournot_three_player_drift,
        diffusion=0.01 * torch.eye(3, dtype=dtype),
        config=DTBConfig(
            step_size=0.005,
            basis_size=8,
            svd_rtol=1e-4,
            jacobian_chunk_size=4,
            derivative_chunk_size=4,
            seed=6,
        ),
    )
    result = method.step(state, 0)
    assert result.state.particles.shape == (4, 3)
    assert result.state.score.shape == (4, 3)
    assert bool(torch.isfinite(result.state.particles).all())
    assert bool(torch.isfinite(result.state.log_density).all())
    assert bool(torch.isfinite(result.state.score).all())


def test_periodic_refit_uses_current_particles_and_resets_block() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    state = gaussian_particle_state(
        12, 2, 0.5, 0.1, device=torch.device("cpu"), dtype=dtype,
        generator=torch.Generator().manual_seed(8),
    )
    model = TangentMLP(2, width=5, depth=1, dtype=dtype)
    method = NeuralDTBGameDynamics(
        model,
        drift=lambda x: 0.5 - x,
        diffusion=0.01 * torch.eye(2, dtype=dtype),
        config=DTBConfig(
            step_size=0.01,
            basis_size=8,
            svd_rtol=1e-5,
            jacobian_chunk_size=12,
            derivative_chunk_size=12,
            seed=9,
            refit_interval=2,
            refit_optimizer_steps=100,
            refit_learning_rate=1e-3,
            refit_batch_size=12,
        ),
    )
    first = method.step(state, 0)
    second = method.step(first.state, 1)

    assert not first.refit_performed
    assert first.steps_in_tangent_block == 1
    assert second.refit_performed
    assert second.refit_reason == "periodic"
    assert second.steps_in_tangent_block == 2
    assert second.refit_rmse_after <= second.refit_rmse_before
    assert method.refit_count == 1
    assert method._block_theta is None


def test_refit_preserves_frozen_mmnn_features() -> None:
    torch.manual_seed(10)
    dtype = torch.float64
    state = gaussian_particle_state(
        10, 2, 0.5, 0.1, device=torch.device("cpu"), dtype=dtype,
        generator=torch.Generator().manual_seed(11),
    )
    model = TangentMMNN(2, width=5, rank=3, depth=2, dtype=dtype)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    method = NeuralDTBGameDynamics(
        model,
        drift=lambda x: 0.5 - x,
        diffusion=0.01 * torch.eye(2, dtype=dtype),
        config=DTBConfig(
            step_size=0.01, basis_size=8, svd_rtol=1e-5,
            jacobian_chunk_size=10, derivative_chunk_size=10, seed=12,
            refit_interval=1, refit_optimizer_steps=5,
            refit_learning_rate=1e-3, refit_batch_size=10,
        ),
    )
    result = method.step(state, 0)

    assert result.refit_performed
    for name, parameter in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(parameter, frozen_before[name])
