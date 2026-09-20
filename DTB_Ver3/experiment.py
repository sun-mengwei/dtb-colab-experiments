"""Complete deterministic and stochastic DTB experiment orchestration.

The notebook-facing API is intentionally small::

    game = CournotGame()
    diffusion = ConstantDiffusion.isotropic(game.dim, amplitude=0.1)
    result = run_experiment(game, ExperimentConfig(dynamics="stochastic"),
                            diffusion=diffusion)

Game equations live in ``games.py``. This module owns initialization, the DTB
time loop, score transport, the automatically selected Euler or
Euler--Maruyama reference, progress reports, and saved numerical results.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dtb import (
    advance_score,
    dtb_step,
    evaluate_dtb_projection,
    evaluate_model,
    flat_parameters,
    tangent_spatial_terms,
)
from .games import Diffusion, Game
from .models import MMNN, ResidualMLP, ResidualMMNN, TangentMLP, count_parameters
from .utils import (
    make_time_grid,
    resolve_device,
    resolve_dtype,
    sample_initial_with_score,
    sliced_wasserstein_distance,
    snapshot_indices,
    to_numpy,
    warmup_cuda,
    write_json,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Numerical, model, stochastic, reference, and output settings."""

    dynamics: str = "deterministic"
    run_reference: bool = True
    reference_integrator: str = "auto"
    reference_seed: int | None = None
    reference_step_size: float | None = None
    particle_count: int = 3000
    initial_law: str = "smoothed_uniform"
    initial_low: float = 0.0
    initial_high: float = 1.0
    smoothing_std: float = 0.02
    gaussian_mean: float = 0.5
    gaussian_std: float = 0.15
    step_size: float = 0.005
    final_time: float = 2.0
    snapshot_times: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    width: int = 16
    rank: int = 12  # Used only by MMNN model kinds.
    depth: int = 2
    activation: str = "tanh"
    model_kind: str = "residual_mlp"
    zero_init_output: bool = False
    basis_size: int = 64
    subset_tangent_selection: str = "resample_each_step"
    tangent_input_mode: str = "current_particles"
    track_network_map: bool = False
    svd_rtol: float = 1e-6
    jacobian_chunk: int = 256
    score_chunk: int = 32
    seed: int = 0
    dtype: str = "float64"
    device: str = "auto"
    cpu_threads: int = 4
    progress_reports: int = 5
    output_dir: str | Path | None = None
    save_outputs: bool = True

    @property
    def stochastic(self) -> bool:
        return self.dynamics == "stochastic"

    def validate(self) -> None:
        if self.dynamics not in {"deterministic", "stochastic"}:
            raise ValueError("dynamics must be 'deterministic' or 'stochastic'")
        if self.reference_integrator not in {
            "auto",
            "euler",
            "rk4",
            "euler_maruyama",
        }:
            raise ValueError(
                "reference_integrator must be 'auto', 'euler', 'rk4', or "
                "'euler_maruyama'"
            )
        if self.stochastic and self.reference_integrator in {"euler", "rk4"}:
            raise ValueError(
                "stochastic dynamics require reference_integrator='auto' or "
                "'euler_maruyama'"
            )
        if not self.stochastic and self.reference_integrator == "euler_maruyama":
            raise ValueError(
                "deterministic dynamics cannot use the Euler--Maruyama reference"
            )
        if self.model_kind not in {
            "mlp",
            "residual_mlp",
            "mmnn",
            "residual_mmnn",
        }:
            raise ValueError(
                "model_kind must be 'mlp', 'residual_mlp', 'mmnn', or "
                "'residual_mmnn'"
            )
        if self.subset_tangent_selection not in {"fixed", "resample_each_step"}:
            raise ValueError(
                "subset_tangent_selection must be 'fixed' or 'resample_each_step'"
            )
        if self.tangent_input_mode not in {
            "current_particles",
            "fixed_initial_labels",
        }:
            raise ValueError(
                "tangent_input_mode must be 'current_particles' or "
                "'fixed_initial_labels'"
            )
        if self.stochastic and self.tangent_input_mode == "fixed_initial_labels":
            raise ValueError(
                "fixed_initial_labels is currently supported for deterministic dynamics"
            )
        if self.track_network_map and self.tangent_input_mode != "fixed_initial_labels":
            raise ValueError(
                "track_network_map requires tangent_input_mode='fixed_initial_labels'"
            )
        if (
            self.particle_count < 1
            or self.basis_size < 1
            or self.jacobian_chunk < 1
            or self.score_chunk < 1
        ):
            raise ValueError(
                "particle_count, basis_size, jacobian_chunk, and score_chunk must be positive"
            )
        if (
            self.width < 1
            or self.depth < 1
            or self.cpu_threads < 1
        ):
            raise ValueError("width, depth, and cpu_threads must be positive")
        if self.model_kind in {"mmnn", "residual_mmnn"}:
            if self.rank < 1:
                raise ValueError("rank must be positive for an MMNN")
            if self.depth < 2:
                raise ValueError("an MMNN requires depth >= 2")
        if self.progress_reports < 0:
            raise ValueError("progress_reports cannot be negative")
        if self.reference_step_size is not None and (
            not np.isfinite(self.reference_step_size)
            or self.reference_step_size <= 0
        ):
            raise ValueError("reference_step_size must be positive and finite")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must lie in [0, 1)")
        if self.gaussian_std <= 0 or self.smoothing_std <= 0:
            raise ValueError("gaussian_std and smoothing_std must be positive")
        if not np.isfinite(self.initial_low) or not np.isfinite(self.initial_high):
            raise ValueError("initial interval bounds must be finite")
        if self.initial_high <= self.initial_low:
            raise ValueError("initial_high must be greater than initial_low")


@dataclass
class ExperimentResult:
    """In-memory arrays, snapshots, diagnostics, and saved-run location."""

    game_name: str
    game_metadata: dict[str, object]
    diffusion_metadata: dict[str, object] | None
    dynamics: str
    reference_method: str
    config: ExperimentConfig
    times: np.ndarray
    projection_times: np.ndarray
    initial_particles: np.ndarray
    initial_score: np.ndarray | None
    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    dtb_final_particles: np.ndarray
    final_score: np.ndarray | None
    reference_final_particles: np.ndarray | None
    dtb_snapshots: dict[float, np.ndarray]
    score_snapshots: dict[float, np.ndarray]
    reference_snapshots: dict[float, np.ndarray]
    network_snapshots: dict[float, np.ndarray]
    projection_error: np.ndarray
    relative_projection_error: np.ndarray
    alpha_norm: np.ndarray
    trajectory_rms_error: np.ndarray
    physical_network_gap: np.ndarray
    network_trajectory_rms_error: np.ndarray
    jacobian_sigma_max: np.ndarray
    jacobian_sigma_min: np.ndarray
    jacobian_condition: np.ndarray
    retained_rank: np.ndarray
    score_rms: np.ndarray
    diffusion_correction_rms: np.ndarray
    selected_indices: np.ndarray
    selected_indices_history: np.ndarray
    final_projection_error: float
    final_relative_projection_error: float
    final_alpha_norm: float
    final_projection_condition: float
    final_projection_rank: int
    elapsed_seconds: float
    final_mean_distance: float | None
    final_paired_rms: float | None
    output_dir: Path | None = None

    @property
    def euler_final_particles(self) -> np.ndarray | None:
        """Compatibility alias for the automatically selected reference."""

        return self.reference_final_particles

    @property
    def euler_snapshots(self) -> dict[float, np.ndarray]:
        """Compatibility alias for the automatically selected reference."""

        return self.reference_snapshots

    def summary(self) -> dict[str, object]:
        """Return the main run facts as a serializable dictionary."""

        return {
            "game": self.game_name,
            "dynamics": self.dynamics,
            "reference_method": self.reference_method,
            "model_kind": self.config.model_kind,
            "subset_tangent_selection": self.config.subset_tangent_selection,
            "tangent_input_mode": self.config.tangent_input_mode,
            "particle_count": int(self.initial_particles.shape[0]),
            "dimension": int(self.initial_particles.shape[1]),
            "step_count": int(len(self.projection_times)),
            "final_time": float(self.times[-1]),
            "elapsed_seconds": self.elapsed_seconds,
            "final_mean_distance": self.final_mean_distance,
            "final_paired_rms": self.final_paired_rms,
            "final_projection_error": self.final_projection_error,
            "final_relative_projection_error": self.final_relative_projection_error,
            "final_alpha_norm": self.final_alpha_norm,
            "final_projection_condition": self.final_projection_condition,
            "final_projection_rank": self.final_projection_rank,
            "final_trajectory_rms_error": (
                None
                if self.trajectory_rms_error.size == 0
                else float(self.trajectory_rms_error[-1])
            ),
            "final_physical_network_gap": (
                None
                if self.physical_network_gap.size == 0
                else float(self.physical_network_gap[-1])
            ),
            "final_network_trajectory_rms_error": (
                None
                if self.network_trajectory_rms_error.size == 0
                else float(self.network_trajectory_rms_error[-1])
            ),
            "final_score_rms": (
                None
                if self.final_score is None
                else float(np.sqrt(np.mean(np.sum(self.final_score**2, axis=1))))
            ),
            "maximum_condition_number": float(np.max(self.jacobian_condition)),
            "output_dir": None if self.output_dir is None else str(self.output_dir),
        }


@dataclass
class StepSizeSweepResult:
    """Runs and the standard numerical table from a step-size sweep."""

    step_sizes: tuple[float, ...]
    runs: dict[float, ExperimentResult]
    records: np.ndarray
    columns: tuple[str, ...]
    metric_name: str
    final_time_rms_error: np.ndarray
    final_relative_projection_error: np.ndarray
    output_dir: Path

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "step_size_sweep.csv"

    @property
    def final_time_rms_csv_path(self) -> Path:
        return self.output_dir / "final_time_rms_vs_step_size.csv"

    @property
    def relative_projection_csv_path(self) -> Path:
        return self.output_dir / "relative_projection_error_vs_step_size.csv"


class DTBExperiment:
    """Reusable runner for one game, one configuration, and optional diffusion."""

    def __init__(
        self,
        game: Game,
        config: ExperimentConfig | None = None,
        *,
        diffusion: Diffusion | None = None,
        model: nn.Module | None = None,
    ) -> None:
        self.game = game
        self.config = config or ExperimentConfig()
        self.diffusion = diffusion
        self.model = model

    def run(self) -> ExperimentResult:
        config = self.config
        config.validate()
        if self.game.dim < 1:
            raise ValueError("game.dim must be positive")
        if config.stochastic and self.diffusion is None:
            raise ValueError("stochastic dynamics require a diffusion object")

        dtype = resolve_dtype(config.dtype)
        device = resolve_device(config.device)
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        warmup_cuda(device, dtype)

        initial_generator = torch.Generator().manual_seed(config.seed)
        initial, initial_score = sample_initial_with_score(
            config.particle_count,
            self.game.dim,
            law=config.initial_law,
            smoothing_std=config.smoothing_std,
            gaussian_mean=config.gaussian_mean,
            gaussian_std=config.gaussian_std,
            dtype=dtype,
            device=device,
            generator=initial_generator,
            uniform_low=config.initial_low,
            uniform_high=config.initial_high,
        )
        particles = initial.detach().clone()
        score = initial_score.detach().clone() if config.stochastic else None
        model = self.model or _make_model(self.game.dim, config, dtype)
        model = model.to(device=device, dtype=dtype)
        theta, structure = flat_parameters(model)
        initial_theta = theta.detach().clone()
        parameter_count = count_parameters(model)
        basis_size = min(config.basis_size, parameter_count)
        initial_network_particles = None
        if config.tangent_input_mode == "fixed_initial_labels":
            initial_network_particles = evaluate_model(
                theta,
                initial,
                model,
                structure,
            ).detach()
            tolerance = 1e-6 if dtype == torch.float32 else 1e-12
            if not torch.allclose(
                initial_network_particles,
                particles,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "fixed_initial_labels requires T_theta0(z)=z; use a "
                    "residual model with zero_init_output=True or supply an "
                    "identity-initialized model"
                )

        basis_generator = torch.Generator().manual_seed(config.seed + 1)
        selected = None
        if config.subset_tangent_selection == "fixed":
            selected = _draw_tangent_subset(
                parameter_count,
                basis_size,
                basis_generator,
                device,
            )

        times = make_time_grid(config.final_time, config.step_size)
        snapshots_by_index = snapshot_indices(times, config.snapshot_times)
        dtb_snapshots = {float(times[0]): to_numpy(particles).copy()}
        score_snapshots = (
            {float(times[0]): to_numpy(score).copy()} if score is not None else {}
        )
        tangent_labels = (
            initial.detach().clone()
            if config.tangent_input_mode == "fixed_initial_labels"
            else None
        )
        network_particles = (
            initial_network_particles
            if config.track_network_map
            else None
        )
        network_snapshots = (
            {float(times[0]): to_numpy(network_particles).copy()}
            if network_particles is not None
            else {}
        )
        projection_error: list[float] = []
        relative_projection_error: list[float] = []
        alpha_norm: list[float] = []
        trajectory_rms_error: list[float] = [0.0] if config.run_reference else []
        physical_network_gap: list[float] = (
            [
                float(
                    (particles - network_particles)
                    .square()
                    .sum(dim=1)
                    .mean()
                    .sqrt()
                    .item()
                )
            ]
            if network_particles is not None
            else []
        )
        network_trajectory_rms_error: list[float] = (
            [
                float(
                    (network_particles - initial)
                    .square()
                    .sum(dim=1)
                    .mean()
                    .sqrt()
                    .item()
                )
            ]
            if network_particles is not None and config.run_reference
            else []
        )
        sigma_max: list[float] = []
        sigma_min: list[float] = []
        condition: list[float] = []
        retained_rank: list[int] = []
        score_rms: list[float] = []
        diffusion_rms: list[float] = []
        selected_history: list[np.ndarray] = []
        reference_particles = initial.detach().clone() if config.run_reference else None
        reference_snapshots = (
            {float(times[0]): to_numpy(reference_particles).copy()}
            if reference_particles is not None
            else {}
        )
        reference_generator = torch.Generator().manual_seed(
            config.seed + 2 if config.reference_seed is None else config.reference_seed
        )

        total_steps = len(times) - 1
        report_steps = _progress_steps(total_steps, config.progress_reports)
        report_index = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        reference_method = _reference_method(config)
        reference_step = (
            "none"
            if not config.run_reference
            else f"{config.reference_step_size or config.step_size:g}"
        )
        print(
            f"DTB run: game={self.game.name}, dynamics={config.dynamics}, "
            f"reference={reference_method}, device={device}, dtype={dtype}, "
            f"particles={config.particle_count}, steps={total_steps}, "
            f"parameters={parameter_count}, model={config.model_kind}, "
            f"subset_size={basis_size}, "
            f"subset_selection={config.subset_tangent_selection}, "
            f"tangent_inputs={config.tangent_input_mode}, "
            f"reference_step={reference_step}",
            flush=True,
        )
        for step in range(total_steps):
            current_time = float(times[step])
            step_size = float(times[step + 1] - times[step])
            if config.subset_tangent_selection == "resample_each_step":
                selected = _draw_tangent_subset(
                    parameter_count,
                    basis_size,
                    basis_generator,
                    device,
                )
            if selected is None:
                raise RuntimeError("tangent subset was not initialized")
            selected_history.append(to_numpy(selected).copy())
            drift = self.game.velocity(particles, current_time)
            _validate_velocity(drift, particles, "game drift")

            if config.stochastic:
                if score is None or self.diffusion is None:
                    raise RuntimeError("stochastic state was not initialized")
                correction = _probability_flow_correction(
                    self.diffusion,
                    particles,
                    score,
                    current_time,
                )
                target = drift - correction
                score_rms.append(float(score.square().sum(dim=1).mean().sqrt().item()))
                diffusion_rms.append(
                    float(correction.square().sum(dim=1).mean().sqrt().item())
                )
            else:
                target = drift
                score_rms.append(float("nan"))
                diffusion_rms.append(float("nan"))

            old_theta = theta
            old_particles = particles
            theta, particles, projection = dtb_step(
                old_theta,
                selected,
                old_particles,
                target,
                model,
                structure,
                step_size=step_size,
                chunk_size=config.jacobian_chunk,
                svd_rtol=config.svd_rtol,
                tangent_inputs=tangent_labels,
            )
            if config.stochastic:
                if score is None:
                    raise RuntimeError("stochastic score was not initialized")
                _, spatial_jacobian, _, gradient_divergence = tangent_spatial_terms(
                    old_theta,
                    selected,
                    projection.alpha,
                    old_particles,
                    model,
                    structure,
                    chunk_size=config.score_chunk,
                )
                score = advance_score(
                    score,
                    spatial_jacobian,
                    gradient_divergence,
                    step_size=step_size,
                )

            if network_particles is not None:
                if tangent_labels is None:
                    raise RuntimeError("network-map tracking has no fixed labels")
                network_particles = evaluate_model(
                    theta,
                    tangent_labels,
                    model,
                    structure,
                ).detach()
                physical_network_gap.append(
                    float(
                        (particles - network_particles)
                        .square()
                        .sum(dim=1)
                        .mean()
                        .sqrt()
                        .item()
                    )
                )

            if reference_particles is not None:
                reference_particles = _advance_reference(
                    self.game,
                    self.diffusion,
                    config,
                    reference_particles,
                    current_time,
                    step_size,
                    reference_generator,
                )
                trajectory_rms_error.append(
                    float(
                        (particles - reference_particles)
                        .square()
                        .sum(dim=1)
                        .mean()
                        .sqrt()
                        .item()
                    )
                )
                if network_particles is not None:
                    network_trajectory_rms_error.append(
                        float(
                            (network_particles - reference_particles)
                            .square()
                            .sum(dim=1)
                            .mean()
                            .sqrt()
                            .item()
                        )
                    )

            projection_error.append(float(projection.rms_residual.item()))
            relative_projection_error.append(float(projection.relative_residual.item()))
            alpha_norm.append(float(torch.linalg.vector_norm(projection.alpha).item()))
            sample_scale = config.particle_count**0.5
            sigma_max.append(float(projection.singular_values[0].item()) * sample_scale)
            sigma_min.append(
                float(
                    projection.singular_values[projection.retained_rank - 1].item()
                )
                * sample_scale
            )
            condition.append(projection.condition_number)
            retained_rank.append(projection.retained_rank)

            state_index = step + 1
            if state_index in snapshots_by_index:
                snapshot_time = snapshots_by_index[state_index]
                dtb_snapshots[snapshot_time] = to_numpy(particles).copy()
                if score is not None:
                    score_snapshots[snapshot_time] = to_numpy(score).copy()
                if reference_particles is not None:
                    reference_snapshots[snapshot_time] = to_numpy(reference_particles).copy()
                if network_particles is not None:
                    network_snapshots[snapshot_time] = to_numpy(network_particles).copy()

            while report_index < len(report_steps) and state_index >= report_steps[report_index]:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                percent = 100.0 * state_index / total_steps
                eta = elapsed * (total_steps - state_index) / state_index
                score_text = "" if score is None else f" | score={score_rms[-1]:.3e}"
                print(
                    f"{percent:5.1f}% | step {state_index}/{total_steps} | "
                    f"t={times[state_index]:g} | residual={projection_error[-1]:.4e}"
                    f"{score_text} | elapsed={elapsed / 60:.1f} min | "
                    f"ETA={eta / 60:.1f} min",
                    flush=True,
                )
                report_index += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        if selected is None:
            raise RuntimeError("tangent subset was not initialized")
        final_time = float(times[-1])
        final_drift = self.game.velocity(particles, final_time)
        _validate_velocity(final_drift, particles, "final game drift")
        if config.stochastic:
            if score is None or self.diffusion is None:
                raise RuntimeError("stochastic final state was not initialized")
            final_target = final_drift - _probability_flow_correction(
                self.diffusion,
                particles,
                score,
                final_time,
            )
        else:
            final_target = final_drift
        final_basis_inputs = particles if tangent_labels is None else tangent_labels
        final_projection = evaluate_dtb_projection(
            theta,
            selected,
            final_basis_inputs,
            final_target,
            model,
            structure,
            chunk_size=config.jacobian_chunk,
            svd_rtol=config.svd_rtol,
        )
        reference_final = (
            None if reference_particles is None else reference_particles.detach()
        )
        final_mean_distance, final_paired_rms = _reference_distances(
            particles,
            reference_final,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        result = ExperimentResult(
            game_name=self.game.name,
            game_metadata=_metadata(self.game),
            diffusion_metadata=(
                None if self.diffusion is None else _metadata(self.diffusion)
            ),
            dynamics=config.dynamics,
            reference_method=reference_method,
            config=config,
            times=times,
            projection_times=times[:-1],
            initial_particles=to_numpy(initial).copy(),
            initial_score=(
                to_numpy(initial_score).copy() if config.stochastic else None
            ),
            initial_parameters=to_numpy(initial_theta).copy(),
            final_parameters=to_numpy(theta).copy(),
            dtb_final_particles=to_numpy(particles).copy(),
            final_score=None if score is None else to_numpy(score).copy(),
            reference_final_particles=(
                None if reference_final is None else to_numpy(reference_final).copy()
            ),
            dtb_snapshots=dtb_snapshots,
            score_snapshots=score_snapshots,
            reference_snapshots=reference_snapshots,
            network_snapshots=network_snapshots,
            projection_error=np.asarray(projection_error),
            relative_projection_error=np.asarray(relative_projection_error),
            alpha_norm=np.asarray(alpha_norm),
            trajectory_rms_error=np.asarray(trajectory_rms_error),
            physical_network_gap=np.asarray(physical_network_gap),
            network_trajectory_rms_error=np.asarray(network_trajectory_rms_error),
            jacobian_sigma_max=np.asarray(sigma_max),
            jacobian_sigma_min=np.asarray(sigma_min),
            jacobian_condition=np.asarray(condition),
            retained_rank=np.asarray(retained_rank),
            score_rms=np.asarray(score_rms),
            diffusion_correction_rms=np.asarray(diffusion_rms),
            selected_indices=to_numpy(selected).copy(),
            selected_indices_history=np.stack(selected_history),
            final_projection_error=float(final_projection.rms_residual.item()),
            final_relative_projection_error=float(
                final_projection.relative_residual.item()
            ),
            final_alpha_norm=float(
                torch.linalg.vector_norm(final_projection.alpha).item()
            ),
            final_projection_condition=final_projection.condition_number,
            final_projection_rank=final_projection.retained_rank,
            elapsed_seconds=elapsed_seconds,
            final_mean_distance=final_mean_distance,
            final_paired_rms=final_paired_rms,
        )
        if config.save_outputs:
            result.output_dir = _save_result(result)
        comparison = (
            "reference disabled"
            if final_paired_rms is None
            else (
                f"DTB/{reference_method} mean distance={final_mean_distance:.4e} | "
                f"paired RMS={final_paired_rms:.4e}"
            )
        )
        print(f"Complete in {elapsed_seconds / 60:.1f} min | {comparison}", flush=True)
        return result


def run_experiment(
    game: Game,
    config: ExperimentConfig | None = None,
    *,
    diffusion: Diffusion | None = None,
    model: nn.Module | None = None,
) -> ExperimentResult:
    """Run one complete DTB/reference comparison."""

    return DTBExperiment(game, config, diffusion=diffusion, model=model).run()


def _make_model(
    dim: int,
    config: ExperimentConfig,
    dtype: torch.dtype,
) -> nn.Module:
    common = {
        "width": config.width,
        "depth": config.depth,
        "activation": config.activation,
        "dtype": dtype,
    }
    if config.model_kind == "residual_mlp":
        return ResidualMLP(
            dim,
            zero_init_output=config.zero_init_output,
            **common,
        )
    if config.model_kind == "residual_mmnn":
        return ResidualMMNN(
            dim,
            width=config.width,
            rank=config.rank,
            depth=config.depth,
            activation=config.activation,
            dtype=dtype,
            zero_init_output=config.zero_init_output,
        )
    if config.model_kind == "mmnn":
        if config.zero_init_output:
            raise ValueError("zero_init_output requires a residual model kind")
        return MMNN(
            dim,
            width=config.width,
            rank=config.rank,
            depth=config.depth,
            d_out=dim,
            activation=config.activation,
            dtype=dtype,
        )
    if config.zero_init_output:
        raise ValueError(
            "zero_init_output is only available for residual_mlp or residual_mmnn"
        )
    return TangentMLP(dim, **common)


def _draw_tangent_subset(
    parameter_count: int,
    subset_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw one sorted parameter-coordinate subset on CPU, then transfer it."""

    selected = torch.randperm(parameter_count, generator=generator)[:subset_size]
    return selected.sort().values.to(device)


def run_step_size_sweep(
    game: Game,
    step_sizes,
    base_config: ExperimentConfig | None = None,
    *,
    diffusion: Diffusion | None = None,
    model: nn.Module | None = None,
    output_dir: str | Path | None = None,
    wasserstein_projections: int = 128,
) -> StepSizeSweepResult:
    """Run matched experiments while changing only the time-step size.

    Every run resets the same seeds, so the initial particles and MLP parameters
    agree. With ``resample_each_step``, each run also starts from the same
    reproducible sequence of random tangent-coordinate subsets. Deterministic
    clouds use paired RMS; stochastic clouds use sliced 2-Wasserstein distance
    because individual Euler--Maruyama particles do not share a deterministic
    pairing with the probability-flow particles.
    """

    config = base_config or ExperimentConfig()
    config.validate()
    if not config.run_reference:
        raise ValueError("a step-size sweep requires run_reference=True")
    values = tuple(float(value) for value in step_sizes)
    if not values or any(value <= 0 for value in values):
        raise ValueError("step_sizes must contain positive values")
    if len(set(values)) != len(values):
        raise ValueError("step_sizes must be unique")

    root = _sweep_output_dir(game, config, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs: dict[float, ExperimentResult] = {}
    for step_size in values:
        tag = f"h_{step_size:.12g}".replace(".", "p").replace("-", "m")
        run_config = replace(
            config,
            step_size=step_size,
            output_dir=root / tag,
        )
        print(f"\nStep-size sweep: h={step_size:g}", flush=True)
        runs[step_size] = run_experiment(
            game,
            run_config,
            diffusion=diffusion,
            model=model,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finest_step = min(values)
    finest_reference = runs[finest_step].reference_final_particles
    if finest_reference is None:
        raise RuntimeError("the finest run did not produce a reference cloud")
    metric_name = (
        "sliced_wasserstein_2" if config.stochastic else "paired_particle_rms"
    )
    records: list[list[float]] = []
    for step_size in values:
        result = runs[step_size]
        reference = result.reference_final_particles
        if reference is None:
            raise RuntimeError(f"run h={step_size:g} did not produce a reference cloud")
        records.append(
            [
                step_size,
                float(len(result.projection_times)),
                _cloud_distance(
                    result.dtb_final_particles,
                    finest_reference,
                    stochastic=config.stochastic,
                    projections=wasserstein_projections,
                    seed=config.seed,
                ),
                _cloud_distance(
                    reference,
                    finest_reference,
                    stochastic=config.stochastic,
                    projections=wasserstein_projections,
                    seed=config.seed,
                ),
                _cloud_distance(
                    result.dtb_final_particles,
                    reference,
                    stochastic=config.stochastic,
                    projections=wasserstein_projections,
                    seed=config.seed,
                ),
                result.final_projection_error,
                result.elapsed_seconds,
            ]
        )
    columns = (
        "step_size",
        "step_count",
        "dtb_vs_finest_reference",
        "reference_vs_finest_reference",
        "dtb_vs_same_step_reference",
        "final_projection_error",
        "elapsed_seconds",
    )
    table = np.asarray(records, dtype=float)
    np.savetxt(
        root / "step_size_sweep.csv",
        table,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    final_rms_values = [runs[step_size].final_paired_rms for step_size in values]
    if any(value is None for value in final_rms_values):
        raise RuntimeError("final-time RMS requires a reference solution for every run")
    final_time_rms_error = np.asarray(final_rms_values, dtype=float)
    if not np.isfinite(final_time_rms_error).all():
        raise FloatingPointError("final-time RMS sweep contains a nonfinite value")
    np.savetxt(
        root / "final_time_rms_vs_step_size.csv",
        np.column_stack((values, final_time_rms_error)),
        delimiter=",",
        header="step_size,final_time_rms_error",
        comments="",
    )
    final_relative_projection_error = np.asarray(
        [runs[step_size].final_relative_projection_error for step_size in values],
        dtype=float,
    )
    if not np.isfinite(final_relative_projection_error).all():
        raise FloatingPointError(
            "relative projection-error sweep contains a nonfinite value"
        )
    np.savetxt(
        root / "relative_projection_error_vs_step_size.csv",
        np.column_stack((values, final_relative_projection_error)),
        delimiter=",",
        header="step_size,final_state_relative_projection_error",
        comments="",
    )
    write_json(
        root / "step_size_sweep.json",
        {
            "game": game.name,
            "dynamics": config.dynamics,
            "reference_method": _reference_method(config),
            "model_kind": config.model_kind,
            "subset_tangent_selection": config.subset_tangent_selection,
            "subset_seed": config.seed + 1,
            "step_sizes": list(values),
            "finest_step": finest_step,
            "cloud_metric": metric_name,
            "final_time_rms_definition": (
                "sqrt(mean_i(||X_DTB_i(T)-X_reference_i(T)||_2^2))"
            ),
            "relative_projection_definition": (
                "||J(theta_T) alpha_T-g(X_T,T)||_2/||g(X_T,T)||_2, "
                "recomputed after the final DTB update"
            ),
            "columns": list(columns),
        },
    )
    return StepSizeSweepResult(
        step_sizes=values,
        runs=runs,
        records=table,
        columns=columns,
        metric_name=metric_name,
        final_time_rms_error=final_time_rms_error,
        final_relative_projection_error=final_relative_projection_error,
        output_dir=root,
    )


def _sweep_output_dir(
    game: Game,
    config: ExperimentConfig,
    requested: str | Path | None,
) -> Path:
    if requested is not None:
        return Path(requested).expanduser().resolve()
    name = f"{game.name}_{config.dynamics}_step_sweep"
    if config.output_dir is not None:
        return Path(config.output_dir).expanduser().resolve().parent / name
    return Path(__file__).resolve().parent / "results" / name


def _cloud_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    stochastic: bool,
    projections: int,
    seed: int,
) -> float:
    if stochastic:
        return sliced_wasserstein_distance(
            first,
            second,
            projections=projections,
            seed=seed,
        )
    if first.shape != second.shape:
        raise ValueError("paired deterministic clouds must have equal shapes")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def _probability_flow_correction(
    diffusion: Diffusion,
    particles: torch.Tensor,
    score: torch.Tensor,
    time_value: float,
) -> torch.Tensor:
    r"""Return ``0.5 * (div(a) + a score)`` for ``a = Sigma Sigma^T``."""

    noise_matrix = diffusion.noise_matrix(particles, time_value)
    if (
        noise_matrix.ndim != 3
        or noise_matrix.shape[0] != particles.shape[0]
        or noise_matrix.shape[1] != particles.shape[1]
        or noise_matrix.shape[2] < 1
    ):
        raise ValueError("diffusion noise matrix must have shape (N, d, brownian_dim)")
    covariance = torch.einsum("nir,njr->nij", noise_matrix, noise_matrix)
    divergence = diffusion.covariance_divergence(particles, time_value)
    if divergence.shape != particles.shape:
        raise ValueError("diffusion covariance divergence must have the particle shape")
    correction = 0.5 * (divergence + torch.einsum("nij,nj->ni", covariance, score))
    _validate_velocity(correction, particles, "probability-flow diffusion correction")
    return correction


def rk4_step(
    game: Game,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
) -> torch.Tensor:
    r"""Advance ``dX/dt = b(X,t)`` by one classical RK4 step.

    The update is ``X+ = X + h*(k1 + 2*k2 + 2*k3 + k4)/6`` with the standard
    four Runge--Kutta stages evaluated through ``game.velocity``.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    with torch.no_grad():
        first = game.velocity(particles, time_value)
        _validate_velocity(first, particles, "RK4 k1")
        second = game.velocity(
            particles + 0.5 * step_size * first,
            time_value + 0.5 * step_size,
        )
        _validate_velocity(second, particles, "RK4 k2")
        third = game.velocity(
            particles + 0.5 * step_size * second,
            time_value + 0.5 * step_size,
        )
        _validate_velocity(third, particles, "RK4 k3")
        fourth = game.velocity(
            particles + step_size * third,
            time_value + step_size,
        )
        _validate_velocity(fourth, particles, "RK4 k4")
        result = particles + (step_size / 6.0) * (
            first + 2.0 * second + 2.0 * third + fourth
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError(f"RK4 state is nonfinite after t={time_value:g}")
    return result.detach()


def rk4_flow(
    game: Game,
    particles: torch.Tensor,
    time_value: float,
    interval: float,
    *,
    maximum_step: float,
) -> torch.Tensor:
    r"""Approximate the interval flow ``Phi_interval(particles)`` by RK4.

    Equal substeps no larger than ``maximum_step`` are used, with the final
    substep adjusted so the requested interval endpoint is reached exactly.
    """

    if interval <= 0 or maximum_step <= 0:
        raise ValueError("interval and maximum_step must be positive")
    substeps = max(1, int(np.ceil(interval / maximum_step - 1e-12)))
    substep_size = interval / substeps
    result = particles.detach().clone()
    for substep in range(substeps):
        result = rk4_step(
            game,
            result,
            time_value + substep * substep_size,
            substep_size,
        )
    return result


def _advance_reference(
    game: Game,
    diffusion: Diffusion | None,
    config: ExperimentConfig,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
    noise_generator: torch.Generator,
) -> torch.Tensor:
    """Advance the selected reference method, optionally with refined substeps."""

    maximum_step = config.reference_step_size
    if _reference_method(config) == "rk4":
        return rk4_flow(
            game,
            particles,
            time_value,
            step_size,
            maximum_step=step_size if maximum_step is None else maximum_step,
        )
    substep_count = (
        1
        if maximum_step is None
        else max(1, int(np.ceil(step_size / maximum_step - 1e-12)))
    )
    substep_size = step_size / substep_count
    next_particles = particles
    for substep in range(substep_count):
        next_particles = _advance_reference_one_step(
            game,
            diffusion,
            config,
            next_particles,
            time_value + substep * substep_size,
            substep_size,
            noise_generator,
        )
    return next_particles


def _advance_reference_one_step(
    game: Game,
    diffusion: Diffusion | None,
    config: ExperimentConfig,
    particles: torch.Tensor,
    time_value: float,
    step_size: float,
    noise_generator: torch.Generator,
) -> torch.Tensor:
    """Advance one selected reference-integrator substep."""

    with torch.no_grad():
        method = _reference_method(config)
        drift = game.velocity(particles, time_value)
        _validate_velocity(drift, particles, "reference drift")
        increment = step_size * drift
        if config.stochastic:
            if diffusion is None:
                raise RuntimeError("stochastic reference has no diffusion")
            noise_matrix = diffusion.noise_matrix(particles, time_value)
            if (
                noise_matrix.ndim != 3
                or noise_matrix.shape[0] != particles.shape[0]
                or noise_matrix.shape[1] != particles.shape[1]
                or noise_matrix.shape[2] < 1
            ):
                raise ValueError(
                    "diffusion noise matrix must have shape (N, d, brownian_dim)"
                )
            brownian = torch.randn(
                (particles.shape[0], noise_matrix.shape[2]),
                dtype=particles.dtype,
                generator=noise_generator,
            ).to(particles.device)
            increment = increment + step_size**0.5 * torch.einsum(
                "nir,nr->ni",
                noise_matrix,
                brownian,
            )
        next_particles = particles + increment
        if not torch.isfinite(next_particles).all():
            method = _reference_method(config)
            raise FloatingPointError(
                f"{method} state is nonfinite after t={time_value:g}"
            )
    return next_particles.detach()


def _reference_method(config: ExperimentConfig) -> str:
    if not config.run_reference:
        return "none"
    if config.reference_integrator != "auto":
        return config.reference_integrator
    return "euler_maruyama" if config.stochastic else "euler"


def _reference_distances(
    particles: torch.Tensor,
    reference: torch.Tensor | None,
) -> tuple[float | None, float | None]:
    if reference is None:
        return None, None
    mean_distance = float(
        torch.linalg.vector_norm(particles.mean(dim=0) - reference.mean(dim=0)).item()
    )
    paired_rms = float((particles - reference).square().sum(dim=1).mean().sqrt().item())
    return mean_distance, paired_rms


def _validate_velocity(
    velocity: torch.Tensor,
    particles: torch.Tensor,
    name: str,
) -> None:
    if velocity.shape != particles.shape:
        raise ValueError(f"{name} must have the particle shape")
    if not torch.isfinite(velocity).all():
        raise FloatingPointError(f"{name} contains a nonfinite value")


def _progress_steps(total: int, reports: int) -> list[int]:
    if reports == 0:
        return []
    return sorted({int(np.ceil(total * index / reports)) for index in range(1, reports + 1)})


def _metadata(value) -> dict[str, object]:
    metadata = getattr(value, "metadata", None)
    if callable(metadata):
        return dict(metadata())
    return {"name": value.name}


def _save_result(result: ExperimentResult) -> Path:
    output = result.config.output_dir
    if output is None:
        output = Path(__file__).resolve().parent / "results" / result.game_name
    folder = Path(output).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "initial_particles.npy", result.initial_particles)
    np.save(folder / "dtb_final_particles.npy", result.dtb_final_particles)
    np.save(folder / "initial_parameters.npy", result.initial_parameters)
    np.save(folder / "final_parameters.npy", result.final_parameters)
    np.save(folder / "subset_tangent_indices.npy", result.selected_indices_history)
    if result.initial_score is not None:
        np.save(folder / "initial_score.npy", result.initial_score)
    if result.final_score is not None:
        np.save(folder / "dtb_final_score.npy", result.final_score)
    if result.reference_final_particles is not None:
        np.save(folder / "reference_final_particles.npy", result.reference_final_particles)
    if result.network_snapshots:
        final_snapshot_time = max(result.network_snapshots)
        np.save(
            folder / "network_final_particles.npy",
            result.network_snapshots[final_snapshot_time],
        )

    snapshot_payload: dict[str, np.ndarray] = {
        "times": np.asarray(sorted(result.dtb_snapshots)),
        "dtb": np.stack(
            [result.dtb_snapshots[time] for time in sorted(result.dtb_snapshots)]
        ),
    }
    if result.reference_snapshots:
        snapshot_payload["reference"] = np.stack(
            [
                result.reference_snapshots[time]
                for time in sorted(result.reference_snapshots)
            ]
        )
    if result.network_snapshots:
        snapshot_payload["network"] = np.stack(
            [result.network_snapshots[time] for time in sorted(result.network_snapshots)]
        )
    if result.score_snapshots:
        snapshot_payload["score"] = np.stack(
            [result.score_snapshots[time] for time in sorted(result.score_snapshots)]
        )
    np.savez_compressed(folder / "snapshots.npz", **snapshot_payload)
    np.savetxt(
        folder / "diagnostics.csv",
        np.column_stack(
            (
                result.projection_times,
                result.projection_error,
                result.relative_projection_error,
                result.alpha_norm,
                result.jacobian_sigma_max,
                result.jacobian_sigma_min,
                result.jacobian_condition,
                result.retained_rank,
                result.score_rms,
                result.diffusion_correction_rms,
            )
        ),
        delimiter=",",
        header=(
            "time,rms_projection_error,relative_projection_error,alpha_norm,sigma_max,"
            "sigma_min_retained,condition_number,retained_rank,score_rms,"
            "diffusion_correction_rms"
        ),
        comments="",
    )
    if result.trajectory_rms_error.size:
        np.savetxt(
            folder / "trajectory_rms_error.csv",
            np.column_stack((result.times, result.trajectory_rms_error)),
            delimiter=",",
            header="time,trajectory_rms_error",
            comments="",
        )
    if result.physical_network_gap.size:
        network_reference_error = (
            result.network_trajectory_rms_error
            if result.network_trajectory_rms_error.size
            else np.full_like(result.physical_network_gap, np.nan)
        )
        np.savetxt(
            folder / "network_state_diagnostics.csv",
            np.column_stack(
                (
                    result.times,
                    result.physical_network_gap,
                    network_reference_error,
                )
            ),
            delimiter=",",
            header="time,physical_network_gap,network_trajectory_rms_error",
            comments="",
        )
    write_json(folder / "config.json", asdict(result.config))
    write_json(folder / "game.json", result.game_metadata)
    if result.diffusion_metadata is not None:
        write_json(folder / "diffusion.json", result.diffusion_metadata)
    result.output_dir = folder
    write_json(folder / "summary.json", result.summary())
    return folder
