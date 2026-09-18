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


class ResidualMLP(nn.Module):
    """Residual map ``T_theta(x) = x + net_theta(x)``.

    The identity skip has no trainable parameters, so its parameter tangent is
    the same as the tangent of ``net_theta``. This lets the experiment use the
    residual-map convention from the comparison notebook while continuing to
    advance the accumulated particles directly.
    """

    def __init__(
        self,
        dim: int,
        *,
        width: int = 16,
        depth: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.net = TangentMLP(
            dim,
            width=width,
            depth=depth,
            activation=activation,
            dtype=dtype,
        ).net
        if zero_init_output:
            output_layer = self.net[-1]
            if not isinstance(output_layer, nn.Linear):
                raise TypeError("the residual MLP output layer must be linear")
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        return particles + self.net(particles)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
