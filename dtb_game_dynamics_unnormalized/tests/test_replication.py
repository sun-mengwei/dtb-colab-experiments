import math

import torch

from game_dtb.games import cournot_duopoly_drift
from game_dtb.runner import diffusion_entry_from_noise, snapshot_schedule


def test_target_cournot_equilibria() -> None:
    equilibria = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    drift = cournot_duopoly_drift(equilibria, b=1.0, mu=2.0)
    assert torch.allclose(drift, torch.zeros_like(drift))


def test_thesis_noise_amplitude_gives_algorithm_diffusion() -> None:
    assert math.isclose(diffusion_entry_from_noise(0.1), 0.01)


def test_figure_snapshot_schedule() -> None:
    schedule = snapshot_schedule("0,0.2,0.4,0.6,0.8,1.0", 50, 0.02)
    assert schedule == {0: 0.0, 10: 0.2, 20: 0.4, 30: 0.6, 40: 0.8, 50: 1.0}
