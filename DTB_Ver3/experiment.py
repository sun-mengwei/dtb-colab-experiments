"""Complete deterministic DTB experiment orchestration.

The notebook-facing API is intentionally small::

    game = CournotGame()
    result = run_experiment(game, ExperimentConfig())

Game equations live in ``games.py``. This module owns initialization, the DTB
time loop, the matched explicit-Euler reference, progress reports, and saved
numerical results.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .dtb import dtb_step, flat_parameters
from .games import Game
from .models import TangentMLP, count_parameters
from .utils import (
    make_time_grid,
    resolve_device,
    resolve_dtype,
    sample_initial_particles,
    snapshot_indices,
    to_numpy,
    warmup_cuda,
    write_json,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Numerical, neural-model, and output settings for one run."""

    particle_count: int = 3000
    initial_law: str = "smoothed_uniform"
    smoothing_std: float = 0.02
    step_size: float = 0.005
    final_time: float = 2.0
    snapshot_times: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    width: int = 16
    depth: int = 2
    activation: str = "tanh"
    basis_size: int = 64
    svd_rtol: float = 1e-6
    jacobian_chunk: int = 256
    seed: int = 0
    dtype: str = "float64"
    device: str = "auto"
    cpu_threads: int = 4
    progress_reports: int = 5
    output_dir: str | Path | None = None
    save_outputs: bool = True

    def validate(self) -> None:
        if self.particle_count < 1 or self.basis_size < 1 or self.jacobian_chunk < 1:
            raise ValueError("particle_count, basis_size, and jacobian_chunk must be positive")
        if self.width < 1 or self.depth < 1 or self.cpu_threads < 1:
            raise ValueError("width, depth, and cpu_threads must be positive")
        if self.progress_reports < 0:
            raise ValueError("progress_reports cannot be negative")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must lie in [0, 1)")


@dataclass
class ExperimentResult:
    """In-memory arrays, snapshots, diagnostics, and saved-run location."""

    game_name: str
    game_metadata: dict[str, object]
    config: ExperimentConfig
    times: np.ndarray
    projection_times: np.ndarray
    initial_particles: np.ndarray
    dtb_final_particles: np.ndarray
    euler_final_particles: np.ndarray
    dtb_snapshots: dict[float, np.ndarray]
    euler_snapshots: dict[float, np.ndarray]
    projection_error: np.ndarray
    relative_projection_error: np.ndarray
    jacobian_sigma_max: np.ndarray
    jacobian_sigma_min: np.ndarray
    jacobian_condition: np.ndarray
    retained_rank: np.ndarray
    selected_indices: np.ndarray
    elapsed_seconds: float
    final_mean_distance: float
    final_paired_rms: float
    output_dir: Path | None = None

    def summary(self) -> dict[str, object]:
        """Return the main run facts as a serializable dictionary."""

        return {
            "game": self.game_name,
            "particle_count": int(self.initial_particles.shape[0]),
            "dimension": int(self.initial_particles.shape[1]),
            "step_count": int(len(self.projection_times)),
            "final_time": float(self.times[-1]),
            "elapsed_seconds": self.elapsed_seconds,
            "final_mean_distance": self.final_mean_distance,
            "final_paired_rms": self.final_paired_rms,
            "final_projection_error": float(self.projection_error[-1]),
            "maximum_condition_number": float(np.max(self.jacobian_condition)),
            "output_dir": None if self.output_dir is None else str(self.output_dir),
        }


class DTBExperiment:
    """Reusable runner for one game and one configuration."""

    def __init__(
        self,
        game: Game,
        config: ExperimentConfig | None = None,
        *,
        model: nn.Module | None = None,
    ) -> None:
        self.game = game
        self.config = config or ExperimentConfig()
        self.model = model

    def run(self) -> ExperimentResult:
        config = self.config
        config.validate()
        if self.game.dim < 1:
            raise ValueError("game.dim must be positive")

        dtype = resolve_dtype(config.dtype)
        device = resolve_device(config.device)
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        warmup_cuda(device, dtype)

        initial_generator = torch.Generator().manual_seed(config.seed)
        initial = sample_initial_particles(
            config.particle_count,
            self.game.dim,
            law=config.initial_law,
            smoothing_std=config.smoothing_std,
            dtype=dtype,
            device=device,
            generator=initial_generator,
        )
        particles = initial.detach().clone()
        model = self.model or TangentMLP(
            self.game.dim,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
            dtype=dtype,
        )
        model = model.to(device=device, dtype=dtype)
        theta, structure = flat_parameters(model)
        parameter_count = count_parameters(model)
        basis_size = min(config.basis_size, parameter_count)

        # The coordinate set is selected once and remains fixed for the run.
        basis_generator = torch.Generator().manual_seed(config.seed + 1)
        selected = torch.randperm(parameter_count, generator=basis_generator)[:basis_size]
        selected = selected.sort().values.to(device)

        times = make_time_grid(config.final_time, config.step_size)
        snapshots_by_index = snapshot_indices(times, config.snapshot_times)
        dtb_snapshots = {float(times[0]): to_numpy(particles).copy()}
        projection_error: list[float] = []
        relative_projection_error: list[float] = []
        sigma_max: list[float] = []
        sigma_min: list[float] = []
        condition: list[float] = []
        retained_rank: list[int] = []

        total_steps = len(times) - 1
        report_steps = _progress_steps(total_steps, config.progress_reports)
        report_index = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        print(
            f"DTB run: game={self.game.name}, device={device}, dtype={dtype}, "
            f"particles={config.particle_count}, steps={total_steps}, "
            f"parameters={parameter_count}, fixed_basis={basis_size}",
            flush=True,
        )
        for step in range(total_steps):
            current_time = float(times[step])
            target = self.game.velocity(particles, current_time)
            if target.shape != particles.shape or not torch.isfinite(target).all():
                raise FloatingPointError("game velocity has an invalid shape or nonfinite value")
            theta, particles, projection = dtb_step(
                theta,
                selected,
                particles,
                target,
                model,
                structure,
                step_size=config.step_size,
                chunk_size=config.jacobian_chunk,
                svd_rtol=config.svd_rtol,
            )
            projection_error.append(float(projection.rms_residual.item()))
            relative_projection_error.append(float(projection.relative_residual.item()))
            sample_scale = config.particle_count**0.5
            sigma_max.append(float(projection.singular_values[0].item()) * sample_scale)
            sigma_min.append(float(projection.singular_values[-1].item()) * sample_scale)
            condition.append(projection.condition_number)
            retained_rank.append(projection.retained_rank)

            state_index = step + 1
            if state_index in snapshots_by_index:
                dtb_snapshots[snapshots_by_index[state_index]] = to_numpy(particles).copy()

            while report_index < len(report_steps) and state_index >= report_steps[report_index]:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                percent = 100.0 * state_index / total_steps
                eta = elapsed * (total_steps - state_index) / state_index
                print(
                    f"{percent:5.1f}% | step {state_index}/{total_steps} | "
                    f"t={times[state_index]:g} | residual={projection_error[-1]:.4e} | "
                    f"elapsed={elapsed / 60:.1f} min | ETA={eta / 60:.1f} min",
                    flush=True,
                )
                report_index += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        euler_final, euler_snapshots = _run_euler_reference(
            self.game,
            initial,
            times,
            snapshots_by_index,
        )
        final_mean_distance = float(
            torch.linalg.vector_norm(particles.mean(dim=0) - euler_final.mean(dim=0)).item()
        )
        final_paired_rms = float(
            (particles - euler_final).square().sum(dim=1).mean().sqrt().item()
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        game_metadata = _game_metadata(self.game)
        result = ExperimentResult(
            game_name=self.game.name,
            game_metadata=game_metadata,
            config=config,
            times=times,
            projection_times=times[:-1],
            initial_particles=to_numpy(initial).copy(),
            dtb_final_particles=to_numpy(particles).copy(),
            euler_final_particles=to_numpy(euler_final).copy(),
            dtb_snapshots=dtb_snapshots,
            euler_snapshots=euler_snapshots,
            projection_error=np.asarray(projection_error),
            relative_projection_error=np.asarray(relative_projection_error),
            jacobian_sigma_max=np.asarray(sigma_max),
            jacobian_sigma_min=np.asarray(sigma_min),
            jacobian_condition=np.asarray(condition),
            retained_rank=np.asarray(retained_rank),
            selected_indices=to_numpy(selected).copy(),
            elapsed_seconds=elapsed_seconds,
            final_mean_distance=final_mean_distance,
            final_paired_rms=final_paired_rms,
        )
        if config.save_outputs:
            result.output_dir = _save_result(result)
        print(
            f"Complete in {elapsed_seconds / 60:.1f} min | "
            f"DTB/Euler mean distance={final_mean_distance:.4e} | "
            f"paired RMS={final_paired_rms:.4e}",
            flush=True,
        )
        return result


def run_experiment(
    game: Game,
    config: ExperimentConfig | None = None,
    *,
    model: nn.Module | None = None,
) -> ExperimentResult:
    """Run one complete DTB/Euler comparison."""

    return DTBExperiment(game, config, model=model).run()


def _run_euler_reference(
    game: Game,
    initial: torch.Tensor,
    times: np.ndarray,
    snapshots_by_index: dict[int, float],
) -> tuple[torch.Tensor, dict[float, np.ndarray]]:
    particles = initial.detach().clone()
    snapshots = {float(times[0]): to_numpy(particles).copy()}
    with torch.no_grad():
        for step in range(len(times) - 1):
            step_size = float(times[step + 1] - times[step])
            particles = particles + step_size * game.velocity(particles, float(times[step]))
            if not torch.isfinite(particles).all():
                raise FloatingPointError(f"Euler state is nonfinite at t={times[step + 1]:g}")
            state_index = step + 1
            if state_index in snapshots_by_index:
                snapshots[snapshots_by_index[state_index]] = to_numpy(particles).copy()
    return particles.detach(), snapshots


def _progress_steps(total: int, reports: int) -> list[int]:
    if reports == 0:
        return []
    return sorted({int(np.ceil(total * index / reports)) for index in range(1, reports + 1)})


def _game_metadata(game: Game) -> dict[str, object]:
    metadata = getattr(game, "metadata", None)
    if callable(metadata):
        return dict(metadata())
    return {"name": game.name, "dim": game.dim}


def _save_result(result: ExperimentResult) -> Path:
    output = result.config.output_dir
    if output is None:
        output = Path(__file__).resolve().parent / "results" / result.game_name
    folder = Path(output).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "initial_particles.npy", result.initial_particles)
    np.save(folder / "dtb_final_particles.npy", result.dtb_final_particles)
    np.save(folder / "euler_final_particles.npy", result.euler_final_particles)
    np.save(folder / "selected_parameter_indices.npy", result.selected_indices)
    np.savez_compressed(
        folder / "snapshots.npz",
        times=np.asarray(sorted(result.dtb_snapshots)),
        dtb=np.stack([result.dtb_snapshots[time] for time in sorted(result.dtb_snapshots)]),
        euler=np.stack([result.euler_snapshots[time] for time in sorted(result.euler_snapshots)]),
    )
    np.savetxt(
        folder / "diagnostics.csv",
        np.column_stack(
            (
                result.projection_times,
                result.projection_error,
                result.relative_projection_error,
                result.jacobian_sigma_max,
                result.jacobian_sigma_min,
                result.jacobian_condition,
                result.retained_rank,
            )
        ),
        delimiter=",",
        header=(
            "time,rms_projection_error,relative_projection_error,sigma_max,"
            "sigma_min,condition_number,retained_rank"
        ),
        comments="",
    )
    write_json(folder / "config.json", asdict(result.config))
    write_json(folder / "game.json", result.game_metadata)
    result.output_dir = folder
    write_json(folder / "summary.json", result.summary())
    return folder
