"""Minimal example showing how to supply a new game drift.

Run from the project root with:

    python -m examples.custom_game
"""

from __future__ import annotations

import torch

from game_dtb import DTBConfig, NeuralDTBGameDynamics, TangentMLP, gaussian_particle_state


# BLOCK A — Define b(x).  This example is a stable, coupled two-player game.
def my_game_drift(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 2:
        raise ValueError("this example has two players")
    centered = x - torch.tensor([0.4, 0.6], device=x.device, dtype=x.dtype)
    drift_1 = -1.2 * centered[:, 0] + 0.25 * centered[:, 1]
    drift_2 = -0.15 * centered[:, 0] - 0.9 * centered[:, 1]
    return torch.stack((drift_1, drift_2), dim=1)


def main() -> None:
    # BLOCK B — Initialize a density whose log density and score are known.
    dtype = torch.float64
    device = torch.device("cpu")
    state = gaussian_particle_state(
        32,
        2,
        mean=0.25,
        std=0.15,
        device=device,
        dtype=dtype,
        generator=torch.Generator().manual_seed(7),
    )

    # BLOCK C — Choose the neural tangent basis and diffusion matrix D.
    model = TangentMLP(dim=2, width=12, depth=2, dtype=dtype)
    diffusion = 0.03 * torch.eye(2, dtype=dtype)
    method = NeuralDTBGameDynamics(
        model,
        drift=my_game_drift,
        diffusion=diffusion,
        config=DTBConfig(step_size=0.01, basis_size=32, seed=7),
    )

    # BLOCK D — Advance X_k(z_i), log rho_k, and the score together.
    for step in range(5):
        result = method.step(state, step)
        state = result.state
        print(
            f"step={step + 1} mean={state.particles.mean(0).tolist()} "
            f"projection_residual={result.diagnostics.relative_residual:.3e}"
        )


if __name__ == "__main__":
    main()
