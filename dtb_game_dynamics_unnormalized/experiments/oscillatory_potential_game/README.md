# Oscillatory potential-game experiment

This experiment applies the existing `game_dtb` Deep Tangent Bundle stepper to
the identical-interest game

```text
Phi(x1,x2) = -lambda/2 (x1^2+x2^2)
             -gamma/2 (x1-x2)^2
             +epsilon/omega [cos(omega*x1)+cos(omega*x2)]
```

with velocity `b=grad(Phi)`. The defaults are `lambda=epsilon=0.5`,
`omega in {2*pi,4*pi,8*pi,16*pi}`, and `gamma in {0,0.2}`.

## Workflow

- `game_dtb/games.py` defines the potential, velocity, and analytic Jacobian.
- `game_dtb/state.py::uniform_box_particle_state` draws one shared
  `U([-1,1]^2)` particle set.
- `game_dtb/models.py` supplies the fixed tangent network. Its initialization
  hash is checked across every case.
- `game_dtb/runner.py::run_dtb_trajectory` repeatedly calls the unchanged
  `NeuralDTBGameDynamics.step` implementation and records the complete DTB and
  truncated-SVD diagnostics.
- `experiment.py` adds the high-accuracy RK4 reference, equilibrium/basin
  analysis, plots, tables, and the controlled eight-case loop.

The deterministic example uses a zero diffusion matrix. Score and log-density
are still transported by the existing DTB stepper, but the target velocity is
exactly the game velocity because `D=0`.

## Run

Run the prepared notebook directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sun-mengwei/dtb-colab-experiments/blob/codex/game-dynamics-dtb/dtb_game_dynamics_unnormalized/notebooks/oscillatory_potential_game_colab.ipynb)

The compact notebook runs one `(gamma, lambda)` case from top to bottom. It
defines the game, initialization, RK4 solver, DTB loop, stability check, and
plots directly in its cells while importing only the existing DTB core. It
shows the projection residual and separate RK4 and DTB particle snapshots;
the multi-case sweep and detailed tables remain available through the CLI.

First run the complete workflow with the small preset:

```bash
python run_oscillatory_potential_game.py \
  --smoke \
  --output-root outputs/oscillatory_potential_game_smoke
```

Then run the documented `N=2000`, `T=1` sweep:

```bash
python run_oscillatory_potential_game.py \
  --output-root outputs/oscillatory_potential_game_full
```

An existing nonempty output root is rejected so previous experiments cannot be
silently overwritten. Increase `--reference-substeps` if the saved/reference
refinement check fails, without changing the DTB step size.

A single game can also be exercised through the original generic CLI:

```bash
python run_game_dynamics.py \
  --game oscillatory --dim 2 \
  --oscillatory-omega 25.132741228718345 \
  --oscillatory-gamma 0.2 \
  --initial-distribution uniform --uniform-low -1 --uniform-high 1 \
  --diffusion-entry 0 --noise-std 0 \
  --output-dir outputs/oscillatory_single
```

## Outputs

Each sweep writes the shared initial particles and effective configuration at
the root, raw case data under `results/`, case and cross-frequency plots under
`figures/`, CSV/Markdown summaries under `tables/`, and an automated numerical
readout in `report.md`. The report labels unreliable basin assignments instead
of forcing a classification.
