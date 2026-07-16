import torch

from game_dtb.algorithm import DTBConfig, NeuralDTBGameDynamics
from game_dtb.models import TangentMLP
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
