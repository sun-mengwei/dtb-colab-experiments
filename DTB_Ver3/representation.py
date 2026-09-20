"""Matched final-tangent representation comparisons for oscillatory games."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from .dtb import flat_parameters, project_velocity, subset_tangent_selection
from .experiment import ExperimentConfig, rk4_flow, run_experiment
from .games import OscillatoryNonpotentialGame
from .models import ResidualMMNN, count_parameters
from .refit import NetworkRefitConfig, RefitDTBConfig, run_refit_dtb
from .utils import resolve_device, resolve_dtype, sample_uniform_box, warmup_cuda


@dataclass(frozen=True)
class RepresentationComparisonConfig:
    """Matched training and validation settings for two parameter update rules."""

    omega_multipliers: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0)
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    training_size: int = 10_000
    validation_size: int = 10_000
    validation_seed: int = 9026
    kappa: float = 1.0
    amplitude: float = 1.0
    step_size: float = 0.001
    final_time: float = 0.2
    reference_step_size: float = 0.000125
    width: int = 12
    rank: int = 12
    depth: int = 3
    activation: str = "tanh"
    random_tangent_size: int = 128
    svd_rtol: float = 1e-8
    jacobian_chunk: int = 512
    refit_interval_fraction: float = 0.10
    refit_learning_rate: float = 1e-3
    refit_maximum_steps: int = 100
    refit_relative_tolerance: float = 0.05
    refit_absolute_tolerance: float = 1e-7
    dtype: str = "float64"
    device: str = "auto"
    progress_reports: int = 4

    def validate(self) -> None:
        """Validate the matched sweep, network, projection, and refit controls."""

        if not self.omega_multipliers or any(value <= 0 for value in self.omega_multipliers):
            raise ValueError("omega_multipliers must contain positive values")
        if not self.seeds:
            raise ValueError("seeds must be nonempty")
        if self.training_size < 1 or self.validation_size < 1:
            raise ValueError("training_size and validation_size must be positive")
        if self.kappa <= 0 or self.amplitude < 0:
            raise ValueError("kappa must be positive and amplitude nonnegative")
        if self.step_size <= 0 or self.final_time <= 0:
            raise ValueError("step_size and final_time must be positive")
        step_count = int(round(self.final_time / self.step_size))
        if not np.isclose(
            step_count * self.step_size,
            self.final_time,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("final_time must be an integer multiple of step_size")
        if self.reference_step_size <= 0:
            raise ValueError("reference_step_size must be positive")
        if self.width < 1 or self.rank < 1 or self.depth < 2:
            raise ValueError("MMNN width/rank must be positive and depth at least two")
        if self.random_tangent_size < 1 or self.jacobian_chunk < 1:
            raise ValueError("random_tangent_size and jacobian_chunk must be positive")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must lie in [0, 1)")
        if not 0 < self.refit_interval_fraction <= 1:
            raise ValueError("refit_interval_fraction must lie in (0, 1]")
        if self.progress_reports < 0:
            raise ValueError("progress_reports cannot be negative")
        NetworkRefitConfig(
            learning_rate=self.refit_learning_rate,
            maximum_steps=self.refit_maximum_steps,
            relative_tolerance=self.refit_relative_tolerance,
            absolute_tolerance=self.refit_absolute_tolerance,
        ).validate()


@dataclass(frozen=True)
class RepresentationComparisonResult:
    """Rows and metadata from a matched direct-versus-refit frequency sweep."""

    config: RepresentationComparisonConfig
    columns: tuple[str, ...]
    records: tuple[tuple[object, ...], ...]
    parameter_count: int


def _make_matched_model(
    config: RepresentationComparisonConfig,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ResidualMMNN:
    """Construct the common identity-initialized residual MMNN for one pair."""

    torch.manual_seed(seed)
    return ResidualMMNN(
        2,
        width=config.width,
        rank=config.rank,
        depth=config.depth,
        activation=config.activation,
        dtype=dtype,
        zero_init_output=True,
    ).to(device)


def _full_representation_metrics(
    model: ResidualMMNN,
    final_parameters: np.ndarray,
    validation_labels: torch.Tensor,
    oscillatory_target: torch.Tensor,
    *,
    jacobian_chunk: int,
    svd_rtol: float,
) -> tuple[float, float, int, float]:
    r"""Project a common target on the full final tangent ``D_theta T_theta``.

    The returned values are relative representation error, coefficient norm,
    retained SVD rank, and retained condition number.
    """

    _, structure = flat_parameters(model)
    theta = torch.as_tensor(
        final_parameters,
        dtype=validation_labels.dtype,
        device=validation_labels.device,
    )
    selected = torch.arange(theta.numel(), device=theta.device)
    _, tangent_matrix = subset_tangent_selection(
        theta,
        selected,
        validation_labels,
        model,
        structure,
        chunk_size=jacobian_chunk,
    )
    projection = project_velocity(
        tangent_matrix,
        oscillatory_target,
        rtol=svd_rtol,
    )
    return (
        float(projection.relative_residual.item()),
        float(torch.linalg.vector_norm(projection.alpha).item()),
        projection.retained_rank,
        projection.condition_number,
    )


def run_representation_comparison(
    config: RepresentationComparisonConfig | None = None,
) -> RepresentationComparisonResult:
    r"""Compare final tangent representation after direct updates and refits.

    For each frequency and seed, both methods use the same training cloud,
    initial MMNN, and ordinary-step random tangent schedule.  Representation is
    evaluated on one common independent validation cloud.  Its labels are
    evolved to time ``T`` by refined RK4, and both full final tangent spaces are
    projected against the same oscillatory target ``q_omega(X_ref(T))``.
    """

    config = config or RepresentationComparisonConfig()
    config.validate()
    dtype = resolve_dtype(config.dtype)
    device = resolve_device(config.device)
    warmup_cuda(device, dtype)
    validation_labels = sample_uniform_box(
        config.validation_size,
        2,
        low=-1.0,
        high=1.0,
        dtype=dtype,
        device=device,
        seed=config.validation_seed,
    )
    refit_optimizer = NetworkRefitConfig(
        learning_rate=config.refit_learning_rate,
        maximum_steps=config.refit_maximum_steps,
        relative_tolerance=config.refit_relative_tolerance,
        absolute_tolerance=config.refit_absolute_tolerance,
    )
    columns = (
        "omega_multiple",
        "omega",
        "seed",
        "method",
        "training_size",
        "validation_size",
        "representation_error",
        "captured_energy",
        "alpha_norm",
        "retained_rank",
        "condition_number",
        "final_trajectory_rms",
        "elapsed_seconds",
        "refit_events",
        "final_refit_rms",
    )
    rows: list[tuple[object, ...]] = []
    parameter_count: int | None = None

    for multiplier in config.omega_multipliers:
        omega = float(multiplier * math.pi)
        game = OscillatoryNonpotentialGame(
            config.kappa,
            config.amplitude,
            omega,
        )
        print(
            f"Building common RK4 validation target for omega/pi={multiplier:g}",
            flush=True,
        )
        validation_state = rk4_flow(
            game,
            validation_labels,
            0.0,
            config.final_time,
            maximum_step=config.reference_step_size,
        )
        validation_target = game.oscillatory_velocity(validation_state)

        for seed in config.seeds:
            base_model = _make_matched_model(
                config,
                seed=seed + 100,
                dtype=dtype,
                device=device,
            )
            direct_model = copy.deepcopy(base_model)
            refit_model = copy.deepcopy(base_model)
            current_parameter_count = count_parameters(base_model)
            if parameter_count is None:
                parameter_count = current_parameter_count
            elif parameter_count != current_parameter_count:
                raise RuntimeError("MMNN parameter count changed across matched runs")
            random_size = min(config.random_tangent_size, current_parameter_count)

            print(
                f"Direct update: omega/pi={multiplier:g}, seed={seed}",
                flush=True,
            )
            direct_config = ExperimentConfig(
                dynamics="deterministic",
                run_reference=True,
                reference_integrator="rk4",
                reference_step_size=config.reference_step_size,
                particle_count=config.training_size,
                initial_law="uniform",
                initial_low=-1.0,
                initial_high=1.0,
                step_size=config.step_size,
                final_time=config.final_time,
                snapshot_times=(0.0, config.final_time),
                width=config.width,
                rank=config.rank,
                depth=config.depth,
                activation=config.activation,
                model_kind="residual_mmnn",
                zero_init_output=True,
                basis_size=random_size,
                subset_tangent_selection="resample_each_step",
                tangent_input_mode="fixed_initial_labels",
                track_network_map=False,
                svd_rtol=config.svd_rtol,
                jacobian_chunk=config.jacobian_chunk,
                seed=seed,
                dtype=config.dtype,
                device=config.device,
                progress_reports=config.progress_reports,
                save_outputs=False,
            )
            direct = run_experiment(game, direct_config, model=direct_model)

            print(
                f"Periodic reset: omega/pi={multiplier:g}, seed={seed}",
                flush=True,
            )
            periodic_config = RefitDTBConfig(
                particle_count=config.training_size,
                initial_low=-1.0,
                initial_high=1.0,
                step_size=config.step_size,
                final_time=config.final_time,
                snapshot_times=(0.0, config.final_time),
                basis_size=random_size,
                svd_rtol=config.svd_rtol,
                jacobian_chunk=config.jacobian_chunk,
                reference_step_size=config.reference_step_size,
                seed=seed,
                dtype=config.dtype,
                device=config.device,
                progress_reports=config.progress_reports,
                refit_interval_fraction=config.refit_interval_fraction,
                refit=refit_optimizer,
            )
            periodic = run_refit_dtb(game, refit_model, periodic_config)
            if not np.array_equal(
                direct.initial_particles,
                periodic.initial_particles,
            ):
                raise RuntimeError("matched methods did not receive the same training cloud")
            if not np.array_equal(
                direct.initial_parameters,
                periodic.initial_parameters,
            ):
                raise RuntimeError("matched methods did not receive the same MMNN")
            for step_index, periodic_selected in enumerate(
                periodic.selected_indices_history
            ):
                if periodic_selected.size == random_size and not np.array_equal(
                    direct.selected_indices_history[step_index],
                    periodic_selected,
                ):
                    raise RuntimeError(
                        "ordinary-step random tangent schedules are not aligned"
                    )

            for method, model, final_parameters, trajectory_rms, elapsed, events, fit in (
                (
                    "direct",
                    direct_model,
                    direct.final_parameters,
                    direct.final_paired_rms,
                    direct.elapsed_seconds,
                    0,
                    float("nan"),
                ),
                (
                    "periodic_refit",
                    refit_model,
                    periodic.final_parameters,
                    periodic.final_paired_rms,
                    periodic.elapsed_seconds,
                    len(periodic.refit_times),
                    float(periodic.refit_rms_after[-1]),
                ),
            ):
                error, alpha_norm, retained_rank, condition = (
                    _full_representation_metrics(
                        model,
                        final_parameters,
                        validation_labels,
                        validation_target,
                        jacobian_chunk=config.jacobian_chunk,
                        svd_rtol=config.svd_rtol,
                    )
                )
                captured_energy = max(0.0, 1.0 - error**2)
                rows.append(
                    (
                        float(multiplier),
                        omega,
                        int(seed),
                        method,
                        config.training_size,
                        config.validation_size,
                        error,
                        captured_energy,
                        alpha_norm,
                        retained_rank,
                        condition,
                        float(trajectory_rms),
                        elapsed,
                        events,
                        fit,
                    )
                )
                print(
                    f"omega/pi={multiplier:g} seed={seed} method={method} "
                    f"E_repr={error:.4e} rank={retained_rank}",
                    flush=True,
                )

    if parameter_count is None:
        raise RuntimeError("comparison produced no runs")
    return RepresentationComparisonResult(
        config=config,
        columns=columns,
        records=tuple(rows),
        parameter_count=parameter_count,
    )
