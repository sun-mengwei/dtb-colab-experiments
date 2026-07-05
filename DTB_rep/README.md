# DTB reproduction: 5-D Allen-Cahn

Reproduction of Section 3.x (5-D Allen-Cahn) from

> Hao Wu, *Deep Tangent Bundle (DTB) Method: a Deep Neural Network
> approach to compute solutions of PDEs*, arXiv:2509.00957.

PDE:
```
  d_t u = nu * Delta(u) + u - u^3 ,   nu = 0.01
  z in [-1, 1]^5,  periodic BCs,  T = 2.0
  u(0, z) = w(z)                       (eqs (51)-(52))
```

Algorithm:
* **Algorithm 3** of the paper -- Forward-Euler DTB:
  `u^{k+1} = u^k + h * J(theta^block; z) * alpha^k` with
  `alpha^k = J(theta^block; z)^+ * F[u^k]`.
* **Proposition 2.4** -- every `L = 20` steps reinitialise by an L^2
  fit `theta_new = argmin ||u^{(j+1)L} - f_{theta_new}||`.

Architecture (`PeriodicMMNN`, `network.py`):
* **Periodic embedding** `PL(z)_{i,j} = cos(pi*z_i + psi_{i,j})`,
  `5 * 40 = 200` features, only `psi` trainable.
* **MMNN** with shape `(w=366, r=25, l=7)`, `h(x) = A sigma(Wx + b) + c`
  per layer with `W, b` frozen and `A, c` trainable.  Activation: GELU
  (paper does not specify; GELU is empirically the only smooth choice
  that keeps the Laplacian bounded across resets; tanh diverges, ReLU
  has identically zero Laplacian which makes `J` rank-deficient).

DTB basis size: `m = 6000` random parameter indices from the
`M = 55617` trainable coordinates, redrawn every block.  Sample size
per step: `N = 20000` MC points uniform in `[-1,1]^5`.

## Files

```
network.py        PeriodicEmbedding + MMNN modules
dtb.py            torch.func-based Jacobian / Laplacian and J-form solver
problem.py        IC w(z), F[u] = nu Lap(u) + u - u^3
run_5d_ac.py      main driver (Algorithm 3 + Prop. 2.4)
visualize.py      reproduces Fig. 5 (snapshots on five 2D hyperplanes)
bench_jac.py      microbenchmark for the Jacobian sweep
bench_svd.py      backend comparison for J-form solve
inspect_*.py      diagnostics on saved snapshots
```

## Running

```bash
pip install -r requirements.txt

# Full paper-scale run (RTX 5080 / 17 GB: ~6 minutes end-to-end).
python run_5d_ac.py --tag paper --fit_steps 6000 --fit_lr 3e-3

# Plot the result (saves PNG into snapshots/run_paper/).
python visualize.py snapshots/run_paper

# Quick smoke (2 blocks, small net, ~10 s):
python run_5d_ac.py --T 0.2 --L 10 --N 2000 --m 1500 \
                    --width 128 --rank 16 --depth 4 \
                    --fit_steps 800 --tag smoke
```

## Notable implementation choices

* **`dtb_basis_matrices` returns `(u, J, Lap u, Lap_J)` in one sweep.**
  We use a single `torch.func.vmap` over MC samples, with nested
  `jacrev` for `J` and `jacrev(_laplacian_one)` for `Lap_J`. Chunking
  along the sample axis caps peak GPU memory.
* **Restricted Jacobian.** `dtb_basis_matrices` only takes derivatives
  with respect to the `m` selected coordinates by patching them into
  the flat trainable vector inside `jacrev`. This avoids materialising
  the full (`N`, `M`) Jacobian.
* **J-form solver.** Default is `torch.linalg.lstsq` on GPU
  (`~0.2 s` for `20000 x 6000` fp32, internally a truncated-SVD
  pseudo-inverse); a CPU/fp64 explicit SVD path (`--solver svd`,
  paper-faithful, ~25 s) is the only alternative. Tikhonov / ridge is
  intentionally *not* offered because the `eps*I` shift biases the
  least-squares solution toward zero and contaminates the
  time-derivative estimate.
* **Periodic reset.** Implemented as Adam on the trainable params
  (`A, c, psi`) against a precomputed snapshot of `u^{(j+1)L}` on
  20 k samples, with cosine LR.  Frozen `W, b` are kept across resets
  to preserve the random-feature manifold.
