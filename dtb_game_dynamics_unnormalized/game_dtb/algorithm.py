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
from .refit import (
    accumulated_tangent_teacher,
    fit_model_to_current_particles,
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
    # A value of zero preserves the original fixed-network behavior.  When
    # positive, one tangent coordinate subset is held for this many physical
    # steps and then compressed into a newly fitted network.
    refit_interval: int = 0
    refit_optimizer_steps: int = 100
    refit_learning_rate: float = 1e-3
    refit_batch_size: int = 256
    # None disables error-triggered refits.  A refit occurs after a step whose
    # relative tangent residual exceeds this value.
    refit_residual_threshold: float | None = None

    def validate(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.basis_size < 1:
            raise ValueError("basis_size must be positive")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must satisfy 0 <= svd_rtol < 1")
        if self.refit_interval < 0:
            raise ValueError("refit_interval must be nonnegative")
        if self.refit_optimizer_steps < 1:
            raise ValueError("refit_optimizer_steps must be positive")
        if self.refit_learning_rate <= 0:
            raise ValueError("refit_learning_rate must be positive")
        if self.refit_batch_size < 1:
            raise ValueError("refit_batch_size must be positive")
        if (
            self.refit_residual_threshold is not None
            and not 0 <= self.refit_residual_threshold <= 1
        ):
            raise ValueError("refit_residual_threshold must lie in [0,1]")


@dataclass(frozen=True)
class StepResult:
    """New particle state plus projection diagnostics."""

    state: ParticleState
    diagnostics: ProjectionDiagnostics
    target_velocity_norm: float
    projected_velocity_norm: float
    alpha_norm: float
    mean_divergence: float
    refit_performed: bool = False
    refit_reason: str = ""
    refit_rmse_before: float = float("nan")
    refit_rmse_after: float = float("nan")
    steps_in_tangent_block: int = 0


class NeuralDTBGameDynamics:
    """Advance particles, log density, and score with Neural--DTB.

    By default the neural parameters remain fixed and each step draws a fresh
    coordinate subset ``S_k``.  Optional periodic/error-triggered refitting
    holds one subset within a block, accumulates its tangent update, fits the
    network to that update on the current particles, and starts a new block.
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

        self._block_theta: torch.Tensor | None = None
        self._block_selected: torch.Tensor | None = None
        self._block_alpha_sum: torch.Tensor | None = None
        self._steps_in_block = 0
        self.refit_count = 0

    @property
    def refitting_enabled(self) -> bool:
        return (
            self.config.refit_interval > 0
            or self.config.refit_residual_threshold is not None
        )

    def _start_tangent_block(self, device: torch.device) -> None:
        """Freeze a linearization point and selected basis for one block."""

        m = min(self.config.basis_size, self.parameter_count)
        self._block_theta = self.theta_flat.detach().clone()
        self._block_selected = torch.randperm(
            self.parameter_count, device=device, generator=self.generator
        )[:m].sort().values
        self._block_alpha_sum = torch.zeros(
            m, device=device, dtype=self.theta_flat.dtype
        )
        self._steps_in_block = 0

    def _reset_tangent_block(self) -> None:
        self._block_theta = None
        self._block_selected = None
        self._block_alpha_sum = None
        self._steps_in_block = 0

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
        if self.refitting_enabled:
            if self._block_theta is None:
                self._start_tangent_block(x_k.device)
            assert self._block_theta is not None
            assert self._block_selected is not None
            assert self._block_alpha_sum is not None
            theta_for_step = self._block_theta
            selected = self._block_selected
        else:
            theta_for_step = self.theta_flat
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
            theta_for_step,
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
                theta_for_step,
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

        # BLOCK 10A — Accumulate the fixed-basis coefficients.  At a block
        # boundary, fit f_theta to f_base + h J_base sum(alpha) using the
        # current particles, then relinearize at the fitted parameters.
        refit_performed = False
        refit_reason = ""
        refit_rmse_before = float("nan")
        refit_rmse_after = float("nan")
        steps_in_tangent_block = 0
        if self.refitting_enabled:
            assert self._block_alpha_sum is not None
            assert self._block_theta is not None
            assert self._block_selected is not None
            self._block_alpha_sum = self._block_alpha_sum + alpha.detach()
            self._steps_in_block += 1
            steps_in_tangent_block = self._steps_in_block
            periodic = (
                self.config.refit_interval > 0
                and self._steps_in_block >= self.config.refit_interval
            )
            excessive_residual = (
                self.config.refit_residual_threshold is not None
                and diagnostics.relative_residual
                > self.config.refit_residual_threshold
            )
            if periodic or excessive_residual:
                reasons = []
                if periodic:
                    reasons.append("periodic")
                if excessive_residual:
                    reasons.append("residual")
                refit_reason = "+".join(reasons)
                targets = accumulated_tangent_teacher(
                    self.model,
                    self._block_theta,
                    self.structure,
                    self._block_selected,
                    self._block_alpha_sum,
                    next_state.particles,
                    step_size=h,
                    jacobian_chunk_size=self.config.jacobian_chunk_size,
                )
                refit = fit_model_to_current_particles(
                    self.model,
                    next_state.particles,
                    targets,
                    optimizer_steps=self.config.refit_optimizer_steps,
                    learning_rate=self.config.refit_learning_rate,
                    batch_size=self.config.refit_batch_size,
                    generator=self.generator,
                )
                self.theta_flat, refreshed_structure = flat_params(self.model)
                self.theta_flat = self.theta_flat.to(
                    self.diffusion.device, self.diffusion.dtype
                )
                if refreshed_structure != self.structure:
                    raise RuntimeError("model parameter structure changed during refit")
                refit_performed = True
                refit_rmse_before = refit.rmse_before
                refit_rmse_after = refit.rmse_after
                self.refit_count += 1
                self._reset_tangent_block()

        return StepResult(
            state=next_state,
            diagnostics=diagnostics,
            target_velocity_norm=float(torch.linalg.norm(target_velocity)),
            projected_velocity_norm=float(torch.linalg.norm(projected_velocity)),
            alpha_norm=float(torch.linalg.norm(alpha)),
            mean_divergence=float(divergence.mean()),
            refit_performed=refit_performed,
            refit_reason=refit_reason,
            refit_rmse_before=refit_rmse_before,
            refit_rmse_after=refit_rmse_after,
            steps_in_tangent_block=steps_in_tangent_block,
        )
