"""
5-D Allen-Cahn initial condition and the RHS operator
F[u] = nu * Lap(u) + u - u^3.

Initial condition from eqs (51)-(52) of arXiv:2509.00957, periodic in
each coordinate (via cos / sin of pi z_i), then linearly mapped to
[-1, 1] via   w(z) = -1 + 2 ( w_tilde(z) + 6 ) / 13.
"""

from __future__ import annotations

import math
import torch


# AC diffusivity used in the paper.
NU = 0.01


def w_tilde(z: torch.Tensor) -> torch.Tensor:
    """The unnormalised initial-condition profile from eq. (51).

    z : (N, 5)  with z_i in [-1, 1].
    """
    pi = math.pi
    z1, z2, z3, z4, z5 = (z[..., 0], z[..., 1], z[..., 2],
                          z[..., 3], z[..., 4])
    c1 = torch.cos(pi * z1)
    c2 = torch.cos(pi * z2)
    c3 = torch.cos(pi * z3)
    c4 = torch.cos(pi * z4)
    c5 = torch.cos(pi * z5)
    s1 = torch.sin(pi * z1)
    s2 = torch.sin(pi * z2)
    s3 = torch.sin(pi * z3)
    s4 = torch.sin(pi * z4)
    s5 = torch.sin(pi * z5)

    out = c1 ** 2                                    # cz1^2
    out = out + s2 ** 3                              # sz2^3
    out = out + 1.5 * (s1 ** 2) * c5                 # 1.5 sz1^2 cz5
    out = out + 3.0 * (1.0 - torch.exp(s2)) / (1.0 + torch.exp(c4))
    out = out + 2.0 * s1 * c3                        # 2 sz1 cz3
    out = out + torch.log(2.0 + c4 * (s1 ** 2)) / torch.exp(c5 + 0.3 * s4)
    out = out + 3.0 * torch.log(3.0 + c2 + s5) / (3.0 + s3)
    return out


def initial_condition(z: torch.Tensor) -> torch.Tensor:
    """w(z) = -1 + 2 * (w_tilde(z) + 6) / 13, see paper eq (52)."""
    return -1.0 + 2.0 * (w_tilde(z) + 6.0) / 13.0


def ac_rhs(u: torch.Tensor, lap_u: torch.Tensor,
           nu: float = NU) -> torch.Tensor:
    """F[u] = nu * Lap(u) + u - u^3."""
    return nu * lap_u + u - u ** 3


def estimate_ic_range(n: int = 200_000, device=None, dtype=torch.float32):
    """Sanity check: verify w(z) lies in [-1, 1] empirically."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = (torch.rand(n, 5, dtype=dtype, device=device) * 2.0 - 1.0)
    w = initial_condition(z)
    return float(w.min()), float(w.max()), float(w.mean()), float(w.std())
