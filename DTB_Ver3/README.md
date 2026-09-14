# DTB Ver3

This directory turns deterministic or stochastic dynamics into a small
interface around the DTB projection code. A complete deterministic run is:

```python
from DTB_Ver3 import CournotGame, ExperimentConfig, run_experiment

game = CournotGame(dim=5, b=2.0, mu=7/4)
result = run_experiment(game, ExperimentConfig())
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

Each run resets the same seed. Deterministic sweeps compare paired particles;
stochastic sweeps compare distributions with sliced 2-Wasserstein distance.
The sweep saves `step_size_sweep.csv` and `step_size_sweep.json` alongside the
individual run folders.

`experiment.py` owns initialization, the fixed tangent-coordinate selection,
the direct particle/parameter updates, score transport, progress reports,
reference integration, and output serialization. `games.py` contains dynamics
and diffusion interfaces; `models.py` contains the MLP; `dtb.py` contains
tangent construction and the truncated-SVD projection; and `utils.py` contains
sampling and plots.

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
X[k+1] = X[k] + h J(theta[k], X[k]) alpha[k]
theta[k+1, selected] = theta[k, selected] + h alpha[k]
```

The selected parameter coordinates are fixed for the whole run. The updated
neural map is not evaluated to replace the accumulated particles, and there
are no resets or refits.
