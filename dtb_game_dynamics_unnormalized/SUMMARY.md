# Technical Summary

## Objective

The code realizes a particle approximation of the distributional game
dynamics

\[
\partial_t\rho=-\nabla\!\cdot(\rho b)
+\tfrac12\nabla\!\cdot(D\nabla\rho),
\qquad
v=b-\tfrac12D\nabla\log\rho.
\]

For fixed reference labels `z_i`, the state stores samples
`x_i^k = X_k(z_i)`, log densities `ell_i^k`, and scores `q_i^k`.

## Discrete Neural–DTB step

At each time step, the implementation:

1. selects `m` scalar parameters from all `M` parameters of a smooth neural
   field `f_theta: R^d -> R^d`;
2. evaluates `J_i = partial f_theta(x_i) / partial theta_S` with shape
   `(d,m)`;
3. builds the unnormalized arrays
   `J_stack in R^(N*d by m)` and `V_stack in R^(N*d)`;
4. retains singular values satisfying `sigma_j > tau*sigma_1` and computes
   `alpha = V_r Sigma_r^(-1) U_r^T V_stack`;
5. defines the projected Eulerian field `u(x)=J_S(x)alpha`;
6. transports `x`, `ell`, and `q` with the supplied first-order formulas.

The score contraction in code is exactly

\[
[\nabla u(x_i)]^Tq_i,
\]

and `grad(div u)` is computed by nested automatic differentiation rather than
finite differences.

## Reuse from the original project

The new package was developed separately from the original files. It adapts:

- the flat parameter representation and functional model evaluation from
  `DTB_rep/dtb.py`;
- the selected-coordinate Jacobian construction used by the original DTB
  basis routine;
- the relative truncated-SVD pseudoinverse pattern from `jform_solve`;
- the two-dimensional Cournot velocity from `DTB_rep/run_game_dtb.py`.

The existing `run_game_dtb.py` is deterministic and differentiates a neural
transport map with respect to parameters. The new implementation follows the
provided TeX algorithm instead: it constructs an Eulerian tangent velocity at
the current particles and includes the diffusion, log-density, and score
updates.

## Important implementation choices

- **No normalization:** stacking is a plain reshape. This is explicitly tested.
- **Restricted Jacobian:** only `m` selected coordinates enter `jacrev`; a full
  `(N,d,M)` tensor is not allocated.
- **Smooth activation:** the default is `tanh` because the method needs second
  spatial derivatives.
- **Fixed neural parameters:** the TeX algorithm does not prescribe a reset or
  parameter update. `theta` defines the tangent dictionary, while `S_k` changes
  each step.
- **Known initial score:** the runner starts from an isotropic Gaussian so both
  the log density and score are exact.
- **Validated diffusion:** the Python API requires a finite, symmetric,
  positive-semidefinite matrix.

## Validation completed

The automated tests check:

- raw stacking without `1/N` or `1/sqrt(N)`;
- recovery of a target lying in the selected tangent span;
- closed-form `u`, `grad u`, `div u`, and `grad(div u)` for a quadratic field;
- the Gaussian log-density and score formulas;
- invariance of the full state when both drift and diffusion are zero.

A two-step end-to-end linear-game run also produced `history.npz` and
`summary.png` with finite diagnostics. These are smoke tests of implementation
consistency, not evidence of convergence for a scientific game model.

## Recommended research checks

Before drawing conclusions from a new game, perform time-step refinement,
particle-count refinement, tangent-basis refinement, SVD-tolerance sensitivity,
and independent score/density validation. Monitor the projection residual,
retained rank, `|alpha|`, score norm, and whether the dynamics respect the
game's admissible strategy space.
