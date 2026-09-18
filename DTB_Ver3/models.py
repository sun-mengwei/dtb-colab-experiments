"""Neural models used as DTB tangent parameterizations."""

from __future__ import annotations

import math

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


class MMNNLayer(nn.Module):
    r"""Matrix-mixing layer ``A sigma(W x + b) + c``.

    The random feature parameters ``W`` and ``b`` remain frozen.  DTB evolves
    only the trainable mixing coefficients ``A`` and ``c``.  This is the MMNN
    parameterization used by the Version 2 oscillatory experiment.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        width: int,
        *,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if d_in < 1 or d_out < 1 or width < 1:
            raise ValueError("d_in, d_out, and width must be positive")
        activations = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "gelu": torch.nn.functional.gelu,
            "sin": torch.sin,
        }
        if activation not in activations:
            raise ValueError(f"unknown activation {activation!r}")

        self.W = nn.Parameter(
            torch.randn(width, d_in, dtype=dtype) / math.sqrt(d_in),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.zeros(width, dtype=dtype),
            requires_grad=False,
        )
        self.A = nn.Parameter(
            torch.randn(d_out, width, dtype=dtype) / math.sqrt(width)
        )
        self.c = nn.Parameter(torch.zeros(d_out, dtype=dtype))
        self._activation = activations[activation]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._activation(inputs @ self.W.T + self.b) @ self.A.T + self.c


class MMNN(nn.Module):
    """Composition of frozen-feature matrix-mixing layers.

    ``width`` is the number of frozen nonlinear features in every layer,
    ``rank`` is the intermediate state dimension, and ``depth`` is the number
    of MMNN layers.
    """

    def __init__(
        self,
        d_in: int,
        *,
        width: int,
        rank: int,
        depth: int,
        d_out: int,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("MMNN depth must be at least 2")
        if rank < 1:
            raise ValueError("MMNN rank must be positive")

        layers: list[MMNNLayer] = [
            MMNNLayer(
                d_in,
                rank,
                width,
                activation=activation,
                dtype=dtype,
            )
        ]
        for _ in range(depth - 2):
            layers.append(
                MMNNLayer(
                    rank,
                    rank,
                    width,
                    activation=activation,
                    dtype=dtype,
                )
            )
        layers.append(
            MMNNLayer(
                rank,
                d_out,
                width,
                activation=activation,
                dtype=dtype,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for layer in self.layers:
            values = layer(values)
        if values.shape[-1] == 1:
            values = values.squeeze(-1)
        return values


class ResidualMMNN(nn.Module):
    """Residual map ``T_theta(x) = x + F_theta(x)`` with an MMNN residual."""

    def __init__(
        self,
        dim: int,
        *,
        width: int = 12,
        rank: int = 12,
        depth: int = 3,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.net = MMNN(
            dim,
            width=width,
            rank=rank,
            depth=depth,
            d_out=dim,
            activation=activation,
            dtype=dtype,
        )
        if zero_init_output:
            final_layer = self.net.layers[-1]
            nn.init.zeros_(final_layer.A)
            nn.init.zeros_(final_layer.c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
