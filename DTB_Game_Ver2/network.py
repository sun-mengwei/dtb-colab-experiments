"""
Network for the 5-D Allen-Cahn DTB experiment of
Wu, "Deep Tangent Bundle (DTB) method", arXiv:2509.00957.

  f_theta(z) = phi_{theta_1}( PL_{theta_2}(z) )

  PL_{theta_2}: [-1,1]^5  ->  R^{5*K}            (periodic embedding)
      [PL(z)]_{i,j} = cos(pi * z_i + psi_{i,j}),   i in [d], j in [K]

  phi_{theta_1}: R^{5K} -> R   is an MMNN of shape (w, r, l).

MMNN layer (Zhang-Shen 2024, "Structured and Balanced ..."):
      h(x) = A * sigma(W x + b) + c
where W, b are *randomly initialised and frozen*, and A, c are trainable.
Composing l layers with intermediate dim r and inner width w gives a
network with parameter shape advertised as (w, r, l).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class PeriodicEmbedding(nn.Module):
    """PL(z)_{i,j} = cos(pi * z_i + psi_{i,j}),  z in R^d, output in R^{d*K}.

    Only psi is trainable; the cos and pi*z parts are deterministic. The
    cos enforces 2-periodicity in every z_i without any boundary loss.
    """

    def __init__(self, dim_in: int = 5, k_per_dim: int = 40,
                 dtype=torch.float32):
        super().__init__()
        self.dim_in = dim_in
        self.k_per_dim = k_per_dim
        # psi initialised uniformly in [0, 2*pi); paper does not specify.
        self.psi = nn.Parameter(
            torch.rand(dim_in, k_per_dim, dtype=dtype) * (2.0 * math.pi)
        )

    @property
    def out_dim(self) -> int:
        return self.dim_in * self.k_per_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, dim_in)  ->  (N, dim_in, k_per_dim)  ->  (N, dim_in*k_per_dim)
        arg = math.pi * z.unsqueeze(-1) + self.psi.unsqueeze(0)
        return torch.cos(arg).reshape(z.shape[0], -1)


class MMNNLayer(nn.Module):
    """h(x) = A * sigma(W x + b) + c.

    W in R^{w x d_in}, b in R^{w}, A in R^{d_out x w}, c in R^{d_out}.
    W, b are frozen (registered as buffers); A, c are trainable.
    """

    def __init__(self, d_in: int, d_out: int, width: int,
                 activation: str = "gelu", dtype=torch.float32):
        super().__init__()
        # Frozen random part. Use He-style init for ReLU-like activations.
        W = torch.randn(width, d_in, dtype=dtype) / math.sqrt(d_in)
        b = torch.zeros(width, dtype=dtype)
        # Trainable linear-combination weights.
        # Init A so that the layer output magnitude stays O(1).
        A = torch.randn(d_out, width, dtype=dtype) / math.sqrt(width)
        c = torch.zeros(d_out, dtype=dtype)

        self.W = nn.Parameter(W, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)
        self.A = nn.Parameter(A)
        self.c = nn.Parameter(c)

        self._act = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "gelu": torch.nn.functional.gelu,
            "sin": torch.sin,
        }[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._act(x @ self.W.T + self.b) @ self.A.T + self.c


class MMNN(nn.Module):
    """Composition of MMNNLayer's with the (w, r, l) shape convention.

      width=w, rank=r, depth=l.

      layer 1 :  d_in   -> r        (W: w x d_in,  A: r x w)
      layer k :  r      -> r        (k = 2..l-1)
      layer l :  r      -> d_out    (A: d_out x w)

    For PDE use d_out = 1.
    """

    def __init__(self, d_in: int, width: int, rank: int, depth: int,
                 d_out: int = 1, activation: str = "gelu",
                 dtype=torch.float32):
        super().__init__()
        layers: List[MMNNLayer] = []
        if depth < 2:
            raise ValueError("MMNN depth must be >= 2.")
        # First layer: d_in -> rank
        layers.append(MMNNLayer(d_in, rank, width, activation, dtype))
        # Middle layers: rank -> rank
        for _ in range(depth - 2):
            layers.append(MMNNLayer(rank, rank, width, activation, dtype))
        # Last layer: rank -> d_out
        layers.append(MMNNLayer(rank, d_out, width, activation, dtype))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lay in self.layers:
            h = lay(h)
        # If d_out == 1, squeeze trailing dim for convenience.
        if h.shape[-1] == 1:
            h = h.squeeze(-1)
        return h


class PeriodicMMNN(nn.Module):
    """Full f_theta = phi_{theta_1} o PL_{theta_2} for the 5-D AC run."""

    def __init__(self, dim_in: int = 5, k_per_dim: int = 40,
                 width: int = 366, rank: int = 25, depth: int = 7,
                 activation: str = "gelu", dtype=torch.float32):
        super().__init__()
        self.pl = PeriodicEmbedding(dim_in, k_per_dim, dtype=dtype)
        self.mmnn = MMNN(self.pl.out_dim, width, rank, depth, d_out=1,
                         activation=activation, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mmnn(self.pl(z))


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
