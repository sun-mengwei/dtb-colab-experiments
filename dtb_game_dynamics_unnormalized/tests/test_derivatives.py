import torch
import torch.nn as nn

from game_dtb.derivatives import tangent_velocity_and_spatial_terms
from game_dtb.parameters import flat_params


class QuadraticFeatureField(nn.Module):
    """f_i(x) = theta_i x_i^2 gives closed-form tangent derivatives."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.7, -0.2], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta * x.square()


def test_spatial_terms_match_closed_form() -> None:
    model = QuadraticFeatureField()
    theta, structure = flat_params(model)
    selected = torch.arange(2)
    alpha = torch.tensor([0.4, -0.3], dtype=torch.float64)
    particles = torch.tensor([[0.2, -0.5], [0.7, 0.1]], dtype=torch.float64)

    velocity, grad_u, divergence, grad_divergence = tangent_velocity_and_spatial_terms(
        theta, selected, alpha, particles, model, structure, chunk_size=1
    )

    expected_velocity = alpha * particles.square()
    expected_grad_u = torch.diag_embed(2.0 * alpha * particles)
    expected_divergence = (2.0 * alpha * particles).sum(dim=1)
    expected_grad_divergence = 2.0 * alpha.expand_as(particles)
    assert torch.allclose(velocity, expected_velocity)
    assert torch.allclose(grad_u, expected_grad_u)
    assert torch.allclose(divergence, expected_divergence)
    assert torch.allclose(grad_divergence, expected_grad_divergence)
