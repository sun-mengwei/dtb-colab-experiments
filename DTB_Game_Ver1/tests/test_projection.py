import torch
import torch.nn as nn

from game_dtb.parameters import flat_params
from game_dtb.projection import (
    selected_parameter_jacobian,
    stack_unnormalized_system,
    truncated_svd_solve,
)


class LinearField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 0.2], [-0.3, 0.8]], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


def test_unnormalized_projection_recovers_tangent_target() -> None:
    model = LinearField()
    theta, structure = flat_params(model)
    selected = torch.arange(theta.numel())
    particles = torch.tensor([[0.2, -0.1], [0.7, 0.3], [-0.4, 0.9]], dtype=torch.float64)
    jacobians = selected_parameter_jacobian(theta, selected, particles, model, structure)

    alpha_true = torch.tensor([0.2, -0.1, 0.05, 0.3, -0.2, 0.4], dtype=torch.float64)
    target = torch.einsum("ndm,m->nd", jacobians, alpha_true)
    stacked_jacobian, stacked_target = stack_unnormalized_system(jacobians, target)
    alpha, diagnostics = truncated_svd_solve(stacked_jacobian, stacked_target, rtol=1e-12)

    assert stacked_jacobian.shape == (particles.numel(), theta.numel())
    assert torch.allclose(stacked_jacobian @ alpha, stacked_target, atol=1e-10)
    assert diagnostics.relative_residual < 1e-10


def test_stacking_has_no_normalization() -> None:
    jacobians = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    target = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    stacked_jacobian, stacked_target = stack_unnormalized_system(jacobians, target)
    assert torch.equal(stacked_jacobian, jacobians.reshape(6, 4))
    assert torch.equal(stacked_target, target.reshape(6))
