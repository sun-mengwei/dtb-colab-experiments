from types import SimpleNamespace

import pytest
import torch

from game_dtb.runner import (
    DTBProgressReporter,
    dtb_progress_milestones,
    projection_error_metrics,
)


def test_progress_milestones_report_every_ten_percent() -> None:
    assert dtb_progress_milestones(500) == (
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
    )
    assert dtb_progress_milestones(7) == (1, 2, 3, 4, 5, 6, 7)


def test_projection_error_metrics_use_particle_vector_rmse() -> None:
    result = SimpleNamespace(
        diagnostics=SimpleNamespace(relative_residual=0.25),
        target_velocity_norm=20.0,
    )
    relative, particle_rmse = projection_error_metrics(result, particle_count=25)
    assert relative == pytest.approx(0.25)
    assert particle_rmse == pytest.approx(1.0)


def test_reporter_prints_only_at_a_milestone(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = DTBProgressReporter(
        total_steps=100,
        step_size=0.002,
        particle_count=25,
        basis_size=32,
        device=torch.device("cpu"),
    )
    result = SimpleNamespace(
        diagnostics=SimpleNamespace(relative_residual=0.25, retained_rank=30),
        target_velocity_norm=20.0,
    )

    assert not reporter.update(completed_step=9, result=result, refit_count=0)
    assert reporter.update(completed_step=10, result=result, refit_count=1)
    assert not reporter.update(completed_step=10, result=result, refit_count=1)
    output = capsys.readouterr().out
    assert "DTB  10%" in output
    assert "step 10/100" in output
    assert "relative=2.500e-01" in output
    assert "particle-RMSE=1.000e+00" in output
    assert "rank=30/32" in output
    assert "refits=1" in output
