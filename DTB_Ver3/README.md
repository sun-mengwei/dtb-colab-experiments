# DTB Ver3

This directory turns deterministic or stochastic dynamics into a small
interface around the DTB projection code. A complete deterministic run is:

```python
from DTB_Ver3 import CournotGame, ExperimentConfig, run_experiment

game = CournotGame(dim=5, b=2.0, mu=7/4)
config = ExperimentConfig(
    model_kind="residual_mlp",
    subset_tangent_selection="resample_each_step",
)
result = run_experiment(game, config)
```

For stochastic dynamics with isotropic noise amplitude `0.1`:

```python
from DTB_Ver3 import ConstantDiffusion

diffusion = ConstantDiffusion.isotropic(dim=game.dim, amplitude=0.1)
config = ExperimentConfig(dynamics="stochastic")
result = run_experiment(game, config, diffusion=diffusion)
```

The stochastic DTB target is the probability-flow velocity

```text
g = drift - 0.5 * (div(Sigma Sigma^T) + Sigma Sigma^T score).
```

The experiment transports the score alongside the accumulated particle map.
The reference is selected automatically: explicit Euler for deterministic
dynamics and Euler--Maruyama for stochastic dynamics. Set
`run_reference=False` to skip it.

A matched step-size sweep is one call:

```python
from DTB_Ver3 import run_step_size_sweep

sweep = run_step_size_sweep(
    game,
    step_sizes=(0.02, 0.01, 0.005, 0.0025),
    base_config=config,
    diffusion=diffusion,
)
```

Each run resets the same particle seed, residual-MLP initialization, and tangent
subset generator. With `subset_tangent_selection="resample_each_step"`, a new
reproducible parameter-coordinate subset is drawn at every DTB iteration, as in
`DTB_Game_Ver2/cournot_5d_b2_mu7_4_mlp_deterministic_dtb.ipynb`. Deterministic
sweeps compare paired particles; stochastic sweeps compare distributions with
sliced 2-Wasserstein distance. The sweep also reports

```text
final_time_rms_error(h)
    = sqrt(mean_i(||X_DTB_i(T; h) - X_reference_i(T; h)||_2^2))
```

and saves it in `final_time_rms_vs_step_size.csv`. The sweep additionally saves
`step_size_sweep.csv`, `step_size_sweep.json`, and the individual run folders.
For stochastic runs this label-paired RMS is descriptive; the sliced
2-Wasserstein distance remains the distributional comparison.

Every run records the time-dependent diagnostics

```text
trajectory_rms_error[k]
    = sqrt(mean_i(||X_DTB_i(t_k) - X_reference_i(t_k)||_2^2))

relative_projection_error[k]
    = ||J_k alpha_k - g_k||_2 / ||g_k||_2

alpha_norm[k] = ||alpha_k||_2.
```

The first quantity is saved in `trajectory_rms_error.csv`; the latter two are
columns in `diagnostics.csv`. The notebooks create
`configuration_diagnostics.png` for the selected configuration. A sweep saves
the last-step relative projection error for each step size in
`relative_projection_error_vs_step_size.csv` and plots it in
`relative_projection_error_vs_step_size.png`. Its projection time is
`T - h`, because the projection is evaluated before the last state update.

`experiment.py` owns initialization, subset tangent selection, direct
particle/parameter updates, score transport, progress reports, reference
integration, and output serialization. `games.py` contains dynamics and
diffusion interfaces; `models.py` contains ordinary/residual MLPs and the MMNN
layers; `dtb.py` contains `subset_tangent_selection` and the truncated-SVD
projection; and `utils.py` contains sampling and plots.

To define a future example without editing the runner:

```python
from DTB_Ver3 import FunctionalGame

def velocity(x, t):
    return -x

game = FunctionalGame(dim=3, name="linear_decay", velocity_fn=velocity)
```

An arbitrary state/time-dependent diffusion supplies its noise matrix and the
divergence of its covariance:

```python
from DTB_Ver3 import FunctionalDiffusion

diffusion = FunctionalDiffusion(
    dim=3,
    noise_fn=lambda x, t: sigma(x, t),                   # shape (N, 3, r)
    covariance_divergence_fn=lambda x, t: div_a(x, t),  # shape (N, 3)
)
```

The uncoupled 10D benchmark is also available without changing the runner:

```python
from DTB_Ver3 import BlockCournotGame

game = BlockCournotGame(block_mus=(7/4, 33/20), block_size=5, b=2.0)
```

The oscillatory DTB example is available as a package game and a dedicated
notebook:

```python
import numpy as np

from DTB_Ver3 import ExperimentConfig, OscillatoryGame

game = OscillatoryGame(
    linear_damping=0.5,
    coupling=0.2,
    epsilon=0.5,
    omega=4 * np.pi,
)
config = ExperimentConfig(
    initial_law="uniform",
    initial_low=-1.0,
    initial_high=1.0,
    model_kind="residual_mmnn",
    width=12,
    rank=12,
    depth=3,
    zero_init_output=True,
    basis_size=338,
    subset_tangent_selection="fixed",
    tangent_input_mode="fixed_initial_labels",
    track_network_map=True,
)
```

Here the game drift is evaluated at the DTB particles while the parameter
tangent is evaluated at immutable initial labels. Network tracking saves the
distinct nonlinear neural map and the DTB/network gap. See
`notebooks/oscillatory_accumulated_map_dtb.ipynb`.

The residual MMNN uses layers `A * activation(W x + b) + c`. Its random
features `W,b` are frozen, so only the 338 trainable `A,c` coordinates belong
to `theta` for the 2D width/rank/depth `12/12/3` configuration.

The built-in neural choices are `mlp`, `residual_mlp`, `mmnn`, and
`residual_mmnn`. `width`, `depth`, and `activation` apply to both families;
`rank` applies only to the MMNN choices. The oscillatory notebook exposes one
`NETWORK_FAMILY = "mmnn"` switch. Its `TANGENT_BASIS_SIZE` setting accepts
`None` for the full trainable basis or a positive integer for a smaller
coordinate subset; no post-run assertion assumes a particular size.
Any vector-valued PyTorch module can also be supplied directly:

```python
result = run_experiment(game, config, model=my_model)
```

Its output must have the same shape as the particles. With
`tangent_input_mode="fixed_initial_labels"`, initialize the supplied map as
the identity on those labels.

The two-player high-frequency non-potential benchmark is implemented by
`OscillatoryNonpotentialGame`:

```python
import numpy as np

from DTB_Ver3 import ExperimentConfig, OscillatoryNonpotentialGame, run_experiment

game = OscillatoryNonpotentialGame(kappa=1.0, amplitude=1.0, omega=16 * np.pi)
config = ExperimentConfig(
    dynamics="deterministic",
    reference_integrator="rk4",
    reference_step_size=0.00025,
    step_size=0.001,
    initial_law="uniform",
    initial_low=-1.0,
    initial_high=1.0,
)
result = run_experiment(game, config)
```

`notebooks/oscillatory_nonpotential_frequency_sweep.ipynb` runs the matched
frequency sweep `omega/pi = (1, 4, 8, 16)`, compares DTB with refined-step
RK4, and saves final RMS, relative final RMS, relative tangent-projection
error, field/cloud plots, and hardest-frequency diagnostics. The game
statement does not prescribe `amplitude` or `kappa`; the notebook exposes
both and uses `1.0` for each by default.

`notebooks/oscillatory_nonpotential_four_hypothesis_sweeps.ipynb` performs
the full diagnostic protocol for four candidate failure mechanisms. It uses
`full_tangent_matrix`, `active_tangent_columns`,
`subset_tangent_selection`, and `project_velocity` directly from
`DTB_Ver3.dtb`; the matched dynamical comparison calls
`run_oscillatory_dynamic_diagnostic` from `DTB_Ver3.diagnostics`. The
notebook contains no local function definitions. Its independent sweeps
cover neural tangent-basis reset/enrichment, training sample size, SVD
cutoff, and time step. The game class also exposes `damping_velocity`,
`oscillatory_velocity`, `velocity_jacobian`, and `symmetric_growth_rate` so
the notebook's mathematical diagnostics share the exact implemented field.

`reference_integrator="auto"` preserves the original behavior: Euler for a
deterministic run and Euler--Maruyama for a stochastic run. Deterministic
runs may choose `"rk4"`; `reference_step_size` then sets the maximum RK4
substep inside every DTB step.

The update implemented by the runner is

```text
X[k+1] = X[k] + h J[S_k](theta[k], B[k]) alpha[k]
theta[k+1, S_k] = theta[k, S_k] + h alpha[k]
```

`B[k]` is `X[k]` for `tangent_input_mode="current_particles"` and the fixed
initial labels for `tangent_input_mode="fixed_initial_labels"`.

By default a fresh random parameter subset is selected at every iteration. Set
`subset_tangent_selection="fixed"` to reuse one subset for the whole run. In
both cases, the updated neural map is not evaluated to replace the accumulated
particles, and there are no resets or refits. The full subset history is saved
as `subset_tangent_indices.npy`.
