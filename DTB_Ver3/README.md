# DTB Ver3

This directory turns a game experiment into a small interface around the DTB
projection code. The complete deterministic run is:

```python
from DTB_Ver3 import CournotGame, ExperimentConfig, run_experiment

game = CournotGame(dim=5, b=2.0, mu=7/4)
result = run_experiment(game, ExperimentConfig())
```

`experiment.py` owns initialization, the fixed tangent-coordinate selection,
the direct particle/parameter updates, progress reports, the matched explicit
Euler reference, and output serialization. `games.py` contains only dynamics;
`models.py` contains the MLP; `dtb.py` contains tangent construction and the
truncated-SVD projection; and `utils.py` contains sampling and plots.

To define a future example without editing the runner:

```python
from DTB_Ver3 import FunctionalGame

def velocity(x, t):
    return -x

game = FunctionalGame(dim=3, name="linear_decay", velocity_fn=velocity)
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
