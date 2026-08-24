"""Small neural-map library for Deep Tangent Bundle experiments.

All models accept tensors with shape ``(..., input_dim)`` and return tensors
with shape ``(..., output_dim)``.  ``ResidualMLP`` is the usual choice for a
pushforward map because it can start at (or very near) the identity.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _activation(name: str) -> type[nn.Module]:
    """Return an activation module class from a short, validated name."""

    choices: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"activation must be one of {tuple(choices)}") from exc


class MLP(nn.Module):
    """Fully connected map with ``depth`` hidden layers.

    Args:
        input_dim: Dimension of an input/reference point.
        output_dim: Dimension of the mapped value.
        width: Neurons per hidden layer.
        depth: Number of hidden layers (at least one).
        activation: ``tanh``, ``relu``, ``gelu``, or ``silu``.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 64,
        depth: int = 3,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width, depth) < 1:
            raise ValueError("input_dim, output_dim, width, and depth must be positive")

        activation_cls = _activation(activation)
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.extend(
                [nn.Linear(in_features, width, dtype=dtype), activation_cls()]
            )
            in_features = width
        layers.append(nn.Linear(in_features, output_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    @property
    def last_layer(self) -> nn.Linear:
        """Expose the output layer for controlled map initialization."""

        layer = self.net[-1]
        assert isinstance(layer, nn.Linear)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    """Pushforward map ``X_theta(z) = z + MLP_theta(z)``.

    ``last_layer_scale=0`` starts exactly at the identity.  A small positive
    value, such as ``1e-3``, starts very close to the identity while keeping
    derivatives with respect to every hidden-layer parameter nonzero.
    """

    def __init__(
        self,
        dim: int,
        width: int = 64,
        depth: int = 3,
        activation: str = "tanh",
        last_layer_scale: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if last_layer_scale < 0:
            raise ValueError("last_layer_scale must be nonnegative")
        self.dim = dim
        self.residual = MLP(dim, dim, width, depth, activation, dtype)

        # Scaling the standard initialization is more useful than replacing it
        # when a near-identity map with a non-degenerate full Jacobian is wanted.
        with torch.no_grad():
            self.residual.last_layer.weight.mul_(last_layer_scale)
            self.residual.last_layer.bias.mul_(last_layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class ResidualBlock(nn.Module):
    """Two-layer hidden residual block used by ``ResidualNetwork``."""

    def __init__(
        self,
        width: int,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        activation_cls = _activation(activation)
        self.block = nn.Sequential(
            nn.Linear(width, width, dtype=dtype),
            activation_cls(),
            nn.Linear(width, width, dtype=dtype),
        )
        self.activation = activation_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualNetwork(nn.Module):
    """Deeper ResNet-style vector map with hidden skip connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 64,
        blocks: int = 3,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width, blocks) < 1:
            raise ValueError("dimensions, width, and blocks must be positive")
        activation_cls = _activation(activation)
        self.input = nn.Sequential(
            nn.Linear(input_dim, width, dtype=dtype), activation_cls()
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, activation, dtype) for _ in range(blocks)]
        )
        self.output = nn.Linear(width, output_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def make_network(kind: str, **kwargs) -> nn.Module:
    """Construct a network by name for configuration-driven experiments."""

    builders: dict[str, Callable[..., nn.Module]] = {
        "mlp": MLP,
        "residual_mlp": ResidualMLP,
        "resnet": ResidualNetwork,
    }
    try:
        return builders[kind.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(f"kind must be one of {tuple(builders)}") from exc


def count_trainable(model: nn.Module) -> int:
    """Count scalar parameters that participate in the DTB tangent space."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "MLP",
    "ResidualBlock",
    "ResidualMLP",
    "ResidualNetwork",
    "count_trainable",
    "make_network",
]
