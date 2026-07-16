"""One-step implementation of the unnormalized Neural--DTB algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .derivatives import tangent_velocity_and_spatial_terms
from .parameters import flat_params
from .projection import (
    ProjectionDiagnostics,
    selected_parameter_jacobian,
    stack_unnormalized_system,
    truncated_svd_solve,
)
from .state import ParticleState

Drift = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DTBConfig:
    """Numerical controls for the supplied discrete algorithm."""

    step_size: float = 0.01
    basis_size: int = 64
    svd_rtol: float = 1e-7
    jacobian_chunk_size: int = 128
    derivative_chunk_size: int = 64
    seed: int = 0

    def validate(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.basis_size < 1:
            raise ValueError("basis_size must be positive")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must satisfy 0 <= svd_rtol < 1")


@dataclass(frozen=True)
class StepResult:
    """New particle state plus projection diagnostics."""

    state: ParticleState
    diagnostics: ProjectionDiagnostics
    target_velocity_norm: float
    projected_velocity_norm: float
    alpha_norm: float
    mean_divergence: float


class NeuralDTBGameDynamics:
    """Advance particles, log density, and score with Neural--DTB.

    The neural parameters define a tangent basis and remain fixed in this
    algorithm.  At each step a fresh coordinate subset ``S_k`` is selected,
    exactly as allowed by the supplied specification.  Only the pushforward
    particle state evolves.
    """

    def __init__(
        self,
        model: nn.Module,
        drift: Drift,
        diffusion: torch.Tensor,
        config: DTBConfig,
    ) -> None:
        config.validate()
        if diffusion.ndim != 2 or diffusion.shape[0] != diffusion.shape[1]:
            raise ValueError("diffusion must be a square (d,d) matrix")
        if not torch.isfinite(diffusion).all():
            raise ValueError("diffusion contains non-finite values")
        if not torch.allclose(diffusion, diffusion.T, rtol=1e-6, atol=1e-8):
            raise ValueError("a diffusion matrix must be symmetric")
        if bool((torch.linalg.eigvalsh(diffusion) < -1e-8).any()):
            raise ValueError("a diffusion matrix must be positive semidefinite")

        self.model = model
        self.drift = drift
        self.diffusion = diffusion
        self.config = config
        self.theta_flat, self.structure = flat_params(model)
        self.theta_flat = self.theta_flat.to(diffusion.device, diffusion.dtype)
        self.parameter_count = self.theta_flat.numel()

        generator_device = diffusion.device.type if diffusion.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(config.seed)

    def step(self, state: ParticleState, step_index: int) -> StepResult:
        """Apply Steps 2--10 of one algorithm iteration.

        ``step_index`` is accepted for logging/reproducibility and makes the
        call site mirror the mathematical index ``k``.
        """

        del step_index
        state.validate()
        x_k = state.particles
        q_k = state.score
        if x_k.shape[1] != self.diffusion.shape[0]:
            raise ValueError("particle dimension and diffusion matrix disagree")
        if x_k.device != self.diffusion.device or x_k.dtype != self.diffusion.dtype:
            raise ValueError("state, model, and diffusion must share device and dtype")

        # BLOCK 2 — Select m tangent coordinates S_k from all M parameters.
        m = min(self.config.basis_size, self.parameter_count)
        selected = torch.randperm(
            self.parameter_count,
            device=x_k.device,
            generator=self.generator,
        )[:m].sort().values

        # BLOCK 3 — Compute v_i^k = b(x_i^k) - 1/2 D q_i^k.
        drift_values = self.drift(x_k)
        if drift_values.shape != x_k.shape:
            raise ValueError("drift must return a tensor with shape (N,d)")
        target_velocity = drift_values - 0.5 * (q_k @ self.diffusion.T)

        # BLOCK 4 — Evaluate J_i^k in R^(d x m), without a full M-Jacobian.
        jacobians = selected_parameter_jacobian(
            self.theta_flat,
            selected,
            x_k,
            self.model,
            self.structure,
            chunk_size=self.config.jacobian_chunk_size,
        )

        # BLOCK 5 — Stack the raw system.  No 1/N or 1/sqrt(N) is applied.
        stacked_jacobian, stacked_velocity = stack_unnormalized_system(
            jacobians, target_velocity
        )

        # BLOCK 6 — Solve alpha_k with the relative truncated-SVD rule.
        alpha, diagnostics = truncated_svd_solve(
            stacked_jacobian,
            stacked_velocity,
            rtol=self.config.svd_rtol,
        )

        # BLOCKS 7–8 — Build u_k=J alpha and its required spatial derivatives.
        projected_velocity, grad_u, divergence, grad_divergence = (
            tangent_velocity_and_spatial_terms(
                self.theta_flat,
                selected,
                alpha,
                x_k,
                self.model,
                self.structure,
                chunk_size=self.config.derivative_chunk_size,
            )
        )

        # BLOCK 9 — Explicit Euler.  Every right-hand side uses the old state.
        h = self.config.step_size
        transported_score = torch.einsum("nji,nj->ni", grad_u, q_k)
        particles_next = x_k + h * projected_velocity
        log_density_next = state.log_density - h * divergence
        score_next = q_k - h * (transported_score + grad_divergence)

        # BLOCK 10 — The labels z_i stay fixed; x_i stores X_k(z_i).
        next_state = ParticleState(
            particles=particles_next.detach(),
            log_density=log_density_next.detach(),
            score=score_next.detach(),
            labels=state.labels,
        )
        next_state.validate()
        return StepResult(
            state=next_state,
            diagnostics=diagnostics,
            target_velocity_norm=float(torch.linalg.norm(target_velocity)),
            projected_velocity_norm=float(torch.linalg.norm(projected_velocity)),
            alpha_norm=float(torch.linalg.norm(alpha)),
            mean_divergence=float(divergence.mean()),
        )
