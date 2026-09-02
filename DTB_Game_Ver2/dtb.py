"""
Core DTB primitives for arXiv:2509.00957.

Notation:
  theta_flat       (M,)               full flat parameter vector (trainable)
  sel              LongTensor (m,)    random index subset, m << M  (the
                                       'l' indices of the paper)
  J_full           (N, M)             d f_theta(z_i) / d theta_j
  J_sel = J_full[:, sel]              (N, m), the DTB basis matrix
  Lap_J_sel        (N, m)             columnwise spatial Laplacians
                                       of the basis (= d Lap f / d theta_j)

Strategy:
  - We hold a single trainable nn.Module `model`. `theta_flat` and
    `structure` are read from it via torch.func machinery.
  - To work on the m-dimensional random sub-basis we *patch* only the
    selected coordinates of the flat vector during jacrev. The fixed
    coordinates are passed as constants and never receive gradients,
    so the returned Jacobian has shape (N, m) directly -- no need to
    materialise the full (N, M) matrix.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev, hessian


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------- flat-param utilities -----------------------

def flat_params(model: nn.Module, only_trainable: bool = True
                ) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]],
                           List[Tuple[str, Tuple[int, ...]]]]:
    """Return (flat trainable tensor, trainable structure, frozen structure).

    Frozen params are kept in the model and used as constants by `u_at`.
    """
    parts: List[torch.Tensor] = []
    structure: List[Tuple[str, Tuple[int, ...]]] = []
    frozen: List[Tuple[str, Tuple[int, ...]]] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            parts.append(p.detach().reshape(-1).clone())
            structure.append((name, tuple(p.shape)))
        else:
            frozen.append((name, tuple(p.shape)))
    flat = torch.cat(parts) if parts else torch.empty(0)
    return flat, structure, frozen

def unflatten(flat: torch.Tensor, structure) -> dict:
    out: dict = {}
    offset = 0
    for name, shape in structure:
        n = 1
        for s in shape:
            n *= s
        out[name] = flat[offset:offset + n].reshape(shape)
        offset += n
    return out


def write_flat_into_model(model: nn.Module, flat: torch.Tensor, structure):
    """Copy a flat trainable vector back into the live nn.Module."""
    with torch.no_grad():
        params_by_name = dict(model.named_parameters())
        offset = 0
        for name, shape in structure:
            n = 1
            for s in shape:
                n *= s
            params_by_name[name].copy_(flat[offset:offset + n].reshape(shape))
            offset += n


# -------------- evaluations through functional_call -----------------

def u_at(theta_flat: torch.Tensor, x: torch.Tensor, model: nn.Module,
         structure) -> torch.Tensor:
    """Evaluate model(x) given flat trainable parameters. Frozen params
    are read from the live module."""
    trainable = unflatten(theta_flat, structure)
    # Merge with the frozen params on the module so functional_call has
    # the full dict it expects.
    full = dict(model.named_parameters())
    full = {k: v for k, v in full.items()}
    full.update(trainable)
    return functional_call(model, full, (x,))


def _u_one(theta_flat: torch.Tensor, x_single: torch.Tensor,
           model: nn.Module, structure) -> torch.Tensor:
    return u_at(theta_flat, x_single.unsqueeze(0), model, structure).squeeze(0)


def _laplacian_one(theta_flat: torch.Tensor, x_single: torch.Tensor,
                   model: nn.Module, structure) -> torch.Tensor:
    """Scalar Laplacian sum_k d^2 u / d x_k^2 at one point."""
    fn = lambda xs: _u_one(theta_flat, xs, model, structure)
    H = hessian(fn)(x_single)  # (d, d)
    return torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)


# --------------- Jacobians, restricted to a sub-basis ---------------

def _u_one_subset(theta_sel: torch.Tensor,
                  theta_frozen_part: torch.Tensor,
                  sel: torch.Tensor, full_size: int,
                  x_single: torch.Tensor,
                  model: nn.Module, structure) -> torch.Tensor:
    """Rebuild the full trainable flat vector with theta_sel patched
    into positions `sel` and the rest filled by `theta_frozen_part`,
    then evaluate u at one point."""
    full = theta_frozen_part.clone()
    full = full.index_copy(0, sel, theta_sel)
    return _u_one(full, x_single, model, structure)


def _lap_one_subset(theta_sel: torch.Tensor,
                    theta_frozen_part: torch.Tensor,
                    sel: torch.Tensor, full_size: int,
                    x_single: torch.Tensor,
                    model: nn.Module, structure) -> torch.Tensor:
    full = theta_frozen_part.clone()
    full = full.index_copy(0, sel, theta_sel)
    return _laplacian_one(full, x_single, model, structure)


def dtb_basis_matrices(theta_flat: torch.Tensor, sel: torch.Tensor,
                       x: torch.Tensor, model: nn.Module, structure,
                       chunk: int = 2000
                       ) -> Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
    """Compute, at flat parameters `theta_flat`, sub-basis `sel`, and
    samples `x` (N, d):

        u_vals    : (N,)       u_theta(x_i)
        J_sel     : (N, m)     d u_theta(x_i) / d theta_{sel}
        lap_vals  : (N,)       Lap_x u_theta(x_i)
        Lap_J_sel : (N, m)     d Lap_x u(x_i) / d theta_{sel}

    We sweep samples in chunks of `chunk` rows to bound peak memory.
    """
    full_size = theta_flat.numel()
    theta_sel = theta_flat[sel].detach().clone()
    theta_frozen_part = theta_flat.detach().clone()

    # Two-arg closures, positional only (works cleanly with vmap).
    def u_fn_one(theta_s, x_single):
        full = theta_frozen_part.clone()
        full = full.index_copy(0, sel, theta_s)
        return _u_one(full, x_single, model, structure)

    def lap_fn_one(theta_s, x_single):
        full = theta_frozen_part.clone()
        full = full.index_copy(0, sel, theta_s)
        return _laplacian_one(full, x_single, model, structure)

    jac_u = jacrev(u_fn_one, argnums=0)
    jac_lap = jacrev(lap_fn_one, argnums=0)

    # vmap over the sample axis.
    u_batched = vmap(u_fn_one, in_dims=(None, 0))
    lap_batched = vmap(lap_fn_one, in_dims=(None, 0))
    J_batched = vmap(jac_u, in_dims=(None, 0))
    LJ_batched = vmap(jac_lap, in_dims=(None, 0))

    N = x.shape[0]
    u_chunks: List[torch.Tensor] = []
    J_chunks: List[torch.Tensor] = []
    lap_chunks: List[torch.Tensor] = []
    LJ_chunks: List[torch.Tensor] = []
    for i in range(0, N, chunk):
        xb = x[i:i + chunk]
        u_chunks.append(u_batched(theta_sel, xb))
        lap_chunks.append(lap_batched(theta_sel, xb))
        J_chunks.append(J_batched(theta_sel, xb))
        LJ_chunks.append(LJ_batched(theta_sel, xb))

    u_vals = torch.cat(u_chunks, dim=0)
    lap_vals = torch.cat(lap_chunks, dim=0)
    J_sel = torch.cat(J_chunks, dim=0)
    Lap_J_sel = torch.cat(LJ_chunks, dim=0)
    return u_vals, J_sel, lap_vals, Lap_J_sel


# --------------- J-form solve --------------------------------------

def jform_solve(J: torch.Tensor, g: torch.Tensor,
                rtol: float = 1e-8, method: str = "lstsq") -> torch.Tensor:
    """alpha = J^+ g (unbiased least squares).

    All backends are *truncated-SVD pseudo-inverses* with the same
    truncation rule: keep singular values sigma > rtol * sigma_max and
    zero the rest.  This is an unbiased rank truncation -- *not* a
    Tikhonov shift.

      method='svd_gpu' (default): GPU SVD in J's native dtype (fp32 on
                       cuda).  Fast (~7 s on 20000x6000), full control
                       over rtol, used for the NGM failure-mode
                       experiments.
      method='svd'   : CPU fp64 SVD with the same truncation rule.
                       Slow (~25 s) but most numerically precise.
      method='lstsq' : torch.linalg.lstsq with rcond=rtol on GPU
                       (cuSOLVER GELSD).  Fastest (~0.2 s).  Pass
                       rtol=None (or negative) to use LAPACK's default
                       rcond = max(M,N)*eps, which on fp32 for our
                       sizes implies a *very lax* cutoff of order
                       1e-3 * sigma_max -- enough to mask the NGM
                       failure mode entirely.  Use rtol=1e-8 for a
                       faithful comparison to the paper's strict solve.

    We deliberately do NOT expose a Tikhonov / ridge backend: adding
    eps*I to J^T J biases the solution toward zero and contaminates
    the time-derivative estimate.
    """
    if method == "svd_gpu":
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        tol = rtol * S[0]
        Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
        return Vh.T @ (Sinv * (U.T @ g))

    if method == "svd":
        Jc = J.detach().to("cpu", dtype=torch.float64)
        gc = g.detach().to("cpu", dtype=torch.float64)
        U, S, Vh = torch.linalg.svd(Jc, full_matrices=False)
        tol = rtol * S[0]
        Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
        alpha = Vh.T @ (Sinv * (U.T @ gc))
        return alpha.to(device=J.device, dtype=J.dtype)

    if method == "lstsq":
        rc = float(rtol) if (rtol is not None and rtol > 0) else None
        sol = torch.linalg.lstsq(J, g.unsqueeze(1), rcond=rc)
        return sol.solution.squeeze(1)

    raise ValueError(
        f"unknown solver method: {method!r} "
        f"(supported: 'svd_gpu', 'svd', 'lstsq'; "
        f"Tikhonov/'ridge' intentionally omitted)"
    )


# --------------- periodic reset (Proposition 2.4) -------------------

def fit_model_to_target(model: nn.Module,
                        target_fn: Callable[[torch.Tensor], torch.Tensor],
                        d_in: int = 5, n_samples: int = 20_000,
                        steps: int = 4000, lr: float = 5e-3,
                        batch_size: int = 4000,
                        verbose: bool = False) -> float:
    """Train all trainable params of `model` so that
       model(z) ~= target_fn(z)   for z ~ U([-1, 1]^d_in).

    Returns the final RMS error on a fresh test batch.
    """
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # One-shot training set (matches the way the paper precomputes a
    # snapshot of u^{jL} on a fixed sample set).
    with torch.no_grad():
        z_train = (torch.rand(n_samples, d_in, dtype=dtype, device=dev)
                   * 2.0 - 1.0)
        y_train = target_fn(z_train).detach()

    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.MSELoss()
    for it in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,), device=dev)
        zb = z_train[idx]
        yb = y_train[idx]
        opt.zero_grad(set_to_none=True)
        pred = model(zb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (it % max(1, steps // 10) == 0):
            print(f"    fit it={it:5d}  mse={loss.item():.3e}")

    # Final test error on a fresh batch.
    with torch.no_grad():
        z_test = (torch.rand(4000, d_in, dtype=dtype, device=dev) * 2.0 - 1.0)
        y_test = target_fn(z_test)
        y_pred = model(z_test)
        rmse = float(torch.sqrt(torch.mean((y_pred - y_test) ** 2)))
    return rmse
