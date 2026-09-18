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

`experiment.py` owns initialization, subset tangent selection, direct
particle/parameter updates, score transport, progress reports, reference
integration, and output serialization. `games.py` contains dynamics and
diffusion interfaces; `models.py` contains ordinary and residual MLPs; `dtb.py`
contains `subset_tangent_selection` and the truncated-SVD projection; and
`utils.py` contains sampling and plots.

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

The update implemented by the runner is

```text
X[k+1] = X[k] + h J[S_k](theta[k], X[k]) alpha[k]
theta[k+1, S_k] = theta[k, S_k] + h alpha[k]
```

By default a fresh random parameter subset is selected at every iteration. Set
`subset_tangent_selection="fixed"` to reuse one subset for the whole run. In
both cases, the updated neural map is not evaluated to replace the accumulated
particles, and there are no resets or refits. The full subset history is saved
as `subset_tangent_indices.npy`.
