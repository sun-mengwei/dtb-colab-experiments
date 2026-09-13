"""Neural models used as DTB tangent parameterizations."""

from __future__ import annotations

import torch
import torch.nn as nn


class TangentMLP(nn.Module):
    """Ordinary vector-valued MLP used to construct parameter tangents."""

    def __init__(
        self,
        dim: int,
        *,
        width: int = 16,
        depth: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1 or width < 1 or depth < 1:
            raise ValueError("dim, width, and depth must be positive")
        activations = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"unknown activation {activation!r}")

        layers: list[nn.Module] = []
        in_features = dim
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width, dtype=dtype))
            layers.append(activations[activation]())
            in_features = width
        layers.append(nn.Linear(in_features, dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        return self.net(particles)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
