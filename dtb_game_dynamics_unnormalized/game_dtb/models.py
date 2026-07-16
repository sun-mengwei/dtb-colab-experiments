"""Neural vector field whose parameter tangents form the DTB basis."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TangentMLP(nn.Module):
    """A smooth map ``f_theta: R^d -> R^d`` used only for its tangents.

    ``tanh`` is the default because the score equation requires second
    spatial derivatives of the tangent velocity.  ReLU should not be used:
    its classical second derivative is zero almost everywhere and undefined
    at the kink.
    """

    def __init__(
        self,
        dim: int,
        width: int = 32,
        depth: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")

        activations = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}")

        layers: List[nn.Module] = []
        input_dim = dim
        for _ in range(depth):
            linear = nn.Linear(input_dim, width, dtype=dtype)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activations[activation]()))
            input_dim = width
        output = nn.Linear(input_dim, dim, dtype=dtype)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_trainable(model: nn.Module) -> int:
    """Number ``M`` of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
