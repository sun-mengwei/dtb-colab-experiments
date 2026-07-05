"""
Neural network models implemented from scratch for small experiments.

The progression is:
  TinyNN: a minimal hand-written one-hidden-layer network
  FCNN: a standard fully connected neural network
  MCNN: one frozen-random-feature layer
  MMNN: stacked frozen-random-feature layers
"""

from __future__ import annotations

import math
from typing import Callable, List

import torch
import torch.nn as nn


def count_trainable(model: nn.Module) -> int
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "sin":
        return torch.sin
    raise ValueError(f"unknown activation {name!r}")


def activation_module(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sin":
        return Sin()
    raise ValueError(f"unknown activation {name!r}")


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class TinyNN(nn.Module):
    """A scalar one-hidden-layer neural network with explicit tensors."""

    def __init__(self, in_dim: int = 1, hidden: int = 16,
                 out_dim: int = 1, activation: str = "tanh"):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden) / math.sqrt(in_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, out_dim) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        self.act = activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(x @ self.w1 + self.b1) @ self.w2 + self.b2
        return y.squeeze(-1) if y.shape[-1] == 1 else y


class FCNN(nn.Module):
    """A conventional fully connected neural network."""

    def __init__(self, in_dim: int = 1, hidden: int = 64, depth: int = 3,
                 out_dim: int = 1, activation: str = "tanh"):
        super().__init__()
        if depth < 1:
            raise ValueError("FCNN depth must be at least 1")

        layers: List[nn.Module] = [nn.Linear(in_dim, hidden),
                                   activation_module(activation)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden),
                           activation_module(activation)])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.squeeze(-1) if y.shape[-1] == 1 else y


class RandomFeatureLayer(nn.Module):
    """h(x) = A sigma(Wx + b) + c21·- with W,b frozen and A,c trainable."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 128,
                 activation: str = "gelu", bias_scale: float = 1.0):
        super().__init__()
        W = torch.randn(width, in_dim) / math.sqrt(in_dim)
        b = bias_scale * torch.randn(width)
        A = torch.randn(out_dim, width) / math.sqrt(width)
        c = torch.zeros(out_dim)

        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.A = nn.Parameter(A)
        self.c = nn.Parameter(c)
        self.act = activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(x @ self.W.T + self.b) @ self.A.T + self.c
        return y


class MCNN(nn.Module):
    """Single-layer random-feature model."""

    def __init__(self, in_dim: int = 1, width: int = 128,
                 out_dim: int = 1, activation: str = "gelu"):
        super().__init__()
        self.layer = RandomFeatureLayer(in_dim, out_dim, width, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        return y.squeeze(-1) if y.shape[-1] == 1 else y


class MMNN(nn.Module):
    """Stacked random-feature model using width, rank, depth notation."""

    def __init__(self, in_dim: int = 1, width: int = 128, rank: int = 16,
                 depth: int = 3, out_dim: int = 1, activation: str = "gelu"):
        super().__init__()
        if depth < 2:
            raise ValueError("MMNN depth must be at least 2")

        layers: List[RandomFeatureLayer] = [
            RandomFeatureLayer(in_dim, rank, width, activation)
        ]
        for _ in range(depth - 2):
            layers.append(RandomFeatureLayer(rank, rank, width, activation))
        layers.append(RandomFeatureLayer(rank, out_dim, width, activation))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h.squeeze(-1) if h.shape[-1] == 1 else h


def build_model(name: str, in_dim: int = 1, tiny_hidden: int = 16,
                hidden: int = 64, fcnn_depth: int = 3, width: int = 128,
                rank: int = 16, mmnn_depth: int = 3,
                activation: str = "tanh",
                random_activation: str = "gelu") -> nn.Module:
    if name == "tiny":
        return TinyNN(in_dim=in_dim, hidden=tiny_hidden,
                      activation=activation)
    if name == "fcnn":
        return FCNN(in_dim=in_dim, hidden=hidden, depth=fcnn_depth,
                    activation=activation)
    if name == "mcnn":
        return MCNN(in_dim=in_dim, width=width,
                    activation=random_activation)
    if name == "mmnn":
        return MMNN(in_dim=in_dim, width=width, rank=rank,
                    depth=mmnn_depth, activation=random_activation)
    raise ValueError(f"unknown model {name!r}")
