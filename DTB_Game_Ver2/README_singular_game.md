# Simple singular-drift DTB example

Open [singular_game_2d_parameter_evolving_dtb.ipynb](singular_game_2d_parameter_evolving_dtb.ipynb)
and run all cells in Jupyter or Colab. The notebook is self-contained and uses
PyTorch, NumPy, and Matplotlib.

It contains the drift `b(x)=(-2/r,-1/r)`, where `r=x1-x2`, followed by one
parameter-evolving DTB loop. The residual map starts at the identity; each step
recomputes its selected Jacobian, solves the particle-normalized SVD ridge
problem, updates the selected parameters, and evaluates the updated map on the
original labels.

Defaults: 512 projection particles, 2048 independent snapshot labels,
`2 -> 16 -> 16 -> 2` tanh map, 128 selected parameters, ridge `1e-6`,
step `0.001`, and final time `0.45`. Initial states are `(c+2*r,c+r)` with
`c` uniform on `[-1,1]` and `r` uniform on `[1,2]`. The singular denominator
is used directly, and steps approaching the diagonal are stopped.

The only displayed diagnostics are:

- Six snapshots with a faint density heatmap and points colored by their initial
  coordinate gap `|x1(0)-x2(0)|`. Each particle keeps its color across time,
  with one shared colorbar (purple = smaller gap, yellow = larger gap).
- RMS projection error versus time.
- The coefficient norm `||alpha(t)||_2` versus time.

For this drift, `dr/dt=-1/r`: larger positive gaps close more slowly. The fixed
point colors make that initial-gap dependence visible in the snapshots.

The saved notebook includes the default run's outputs. Checks for the drift,
normalized solve, actual map update, and rejected steps can be run with:

```bash
python -m unittest test_singular_game -v
```
