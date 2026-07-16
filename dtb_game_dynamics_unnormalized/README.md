# Unnormalized Neural–DTB for Distributional Game Dynamics

This standalone project implements the algorithm in
`dtb_game_dynamics_unnormalized_dimensions.tex`. It transports particles,
their log density, and their score under a game drift plus diffusion:

\[
v_k(x_i)=b(x_i)-\tfrac12Dq_i,
\qquad
u_k(x)=J_{k,S_k}(x)\alpha_k.
\]

The coefficient is the relative truncated-SVD solution of the **unnormalized**
stacked system. No `1/N` or `1/sqrt(N)` factor is applied.

The implementation is separate from the original Deep Tangent Bundle code.
It reuses selected ideas and small utilities from
[`DTB_rep/dtb.py`](https://github.com/sun-mengwei/dtb-colab-experiments/blob/main/DTB_rep/dtb.py)
and the Cournot drift from
[`DTB_rep/run_game_dtb.py`](https://github.com/sun-mengwei/dtb-colab-experiments/blob/main/DTB_rep/run_game_dtb.py),
but no original file is changed.

## What is implemented

- A smooth neural field `f_theta: R^d -> R^d`.
- Random selected parameter directions `S_k`, with `m <= M`.
- Restricted per-particle Jacobians `J_i` without constructing the full
  `(N,d,M)` Jacobian.
- The raw stacked system `J_stack` of shape `(N*d,m)`.
- Explicit relative truncated SVD with `sigma_j > tau*sigma_1`.
- Particle, log-density, and score Euler updates.
- Exact automatic differentiation of `grad u`, `div u`, and
  `grad(div u)`.
- Linear-quadratic and nonlinear Cournot example games.
- Tests for the projection, derivatives, initialization, and full step.

## Code layout

```text
game_dtb/
  algorithm.py       numbered Blocks 2–10 and the Euler update
  derivatives.py     grad u, div u, and grad(div u)
  games.py           example game drifts b(x)
  models.py          smooth vector-valued TangentMLP
  parameters.py      flat-parameter helpers adapted from original DTB
  projection.py      restricted Jacobian + unnormalized SVD system
  runner.py          experiment loop, saved arrays, and diagnostic plot
  state.py           particles, log density, score, and Gaussian init
examples/
  custom_game.py     instructional custom-drift example
tests/                mathematical and integration tests
run_game_dynamics.py command-line entry point
SUMMARY.md            method and implementation summary
COLAB_TUTORIAL.md     step-by-step Google Colab instructions
notebooks/            ready-to-upload Colab notebook
```

## Quick start

Create an isolated Python environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

Run a small linear-quadratic game:

```bash
python run_game_dynamics.py \
  --game linear \
  --particles 64 \
  --steps 10 \
  --basis-size 64 \
  --output-dir outputs/linear_smoke
```

Run the nonlinear Cournot game:

```bash
python run_game_dynamics.py \
  --game cournot \
  --particles 128 \
  --steps 25 \
  --step-size 0.005 \
  --basis-size 128 \
  --svd-rtol 1e-5 \
  --diffusion 0.03 \
  --dtype float32 \
  --device auto \
  --output-dir outputs/cournot
```

Each run creates:

- `config.json`: arguments, resolved device, and dimensions;
- `history.npz`: particles, score, log density, means, ranks, and residuals;
- `summary.png`: initial/final particles and mean-strategy evolution.

## Algorithm-to-code map

| Algorithm step | Implementation |
|---|---|
| 1. Initialize `x`, `ell`, `q` | `state.gaussian_particle_state`; Block 1 in `runner.py` |
| 2. Select `S_k` | Block 2 in `NeuralDTBGameDynamics.step` |
| 3. Form game velocity | Block 3 in `algorithm.py` |
| 4. Evaluate selected Jacobians | `projection.selected_parameter_jacobian` |
| 5. Stack without normalization | `projection.stack_unnormalized_system` |
| 6. Truncated-SVD solve | `projection.truncated_svd_solve` |
| 7. Evaluate `u=J alpha` | `derivatives.tangent_velocity_and_spatial_terms` |
| 8. Spatial derivatives | same function, using nested `jacrev` |
| 9. Euler updates | Block 9 in `algorithm.py` |
| 10. Update sampled map | `ParticleState.particles`; labels `z_i` remain fixed |

At the particle level, storing the fixed labels and evolved particles means
`particles[i] = X_k(labels[i])`. The implementation does not construct a
closed-form off-sample representation of the composed map `X_k`.

## Supplying another game

A drift must accept `(N,d)` particles and return an `(N,d)` tensor:

```python
def my_drift(x):
    return -x  # replace with simultaneous player-gradient or response dynamics
```

Pass it directly to `NeuralDTBGameDynamics`. See
[`examples/custom_game.py`](examples/custom_game.py) for a complete commented
example.

For a non-isotropic diffusion, call the Python API with a symmetric positive
semidefinite `(d,d)` tensor. The CLI uses `D = diffusion * I`.

## Choosing numerical settings

- Start with `N=32–64`, `m=32–64`, and 5–10 steps.
- Reduce `h` if particles, score norms, or projection coefficients grow fast.
- Increase `m` if the projection residual stays high, subject to memory.
- Tightening `svd_rtol` retains weaker tangent directions and may sharply
  increase `|alpha|`. Start around `1e-5` for float32 experiments and perform
  a tolerance sweep before interpreting results.
- Use `tanh`, `gelu`, or `silu`. Do not use ReLU because the score equation
  needs second spatial derivatives.
- `grad(div u)` requires nested differentiation and is the main computational
  cost. Scale `N` and `m` gradually on Colab.

## Scope and limitations

- The scheme is explicit Euler and has no adaptive time-step controller.
- Strategies are unconstrained. Add a mathematically justified coordinate
  transform or boundary treatment for simplex/box-constrained games; naive
  clipping is inconsistent with the transported score equation.
- The supplied algorithm does not specify a neural reset/refit rule, so
  `theta` stays fixed while `S_k` is redrawn. This differs from the periodic
  reset used by the original Allen–Cahn DTB experiment.
- The transported score and log density are first-order discretizations. They
  should be monitored for long runs.

For Google Colab, continue with [COLAB_TUTORIAL.md](COLAB_TUTORIAL.md).
