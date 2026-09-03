For $N$ particles in $d$ dimensions and an $m$-vector selected tangent basis,
let $x_{k,j}=X_k(z_j)$ and $J_{k,j}^{S}=\partial_{\theta_S}f_{\theta_0}(x_{k,j})$.
Stack the particle velocities as
$\mathbf{g}_k=\operatorname{col}_{j=1}^{N}b(x_{k,j})$ and
$\mathbf{u}_k=\operatorname{col}_{j=1}^{N}(J_{k,j}^{S}\alpha_k)$, where
$\alpha_k\in\mathbb{R}^{m}$.
Distances use every coordinate and the supplied reference set $\mathcal{E}$:

$$d_{k,j}=\min_{e\in\mathcal{E}}\lVert x_{k,j}-e\rVert_2.$$

**State metrics**

| Metric | Mathematical definition | First: $t=0$ | Last: $t=1$ |
| :--- | :--- | ---: | ---: |
| Game velocity (RMS) | $v_k^b=\frac{\lVert\mathbf{g}_k\rVert_2}{\sqrt{N}}$ | $6.463\times 10^{-1}$ | $9.785\times 10^{-2}$ |
| Median equilibrium distance | $d_k^{50}=\operatorname{median}_{j}\,d_{k,j}$ | $2.584\times 10^{-1}$ | $1.492\times 10^{-1}$ |
| 90th percentile equilibrium distance | $d_k^{90}=Q_{0.9}(\{d_{k,j}\}_{j=1}^{N})$ | $4.865\times 10^{-1}$ | $2.179\times 10^{-1}$ |
| Minimum coordinate | $x_k^{\min}=\min_{j,i}x_{k,j,i}$ | $1.678\times 10^{-4}$ | $-7.478\times 10^{-2}$ |
| Fraction of negative coordinates | $f_k^-=\frac{1}{Nd}\sum_{j=1}^{N}\sum_{i=1}^{d}\mathbf{1}_{\{x_{k,j,i}<0\}}$ | $0$ | $1.215\times 10^{-2}$ |

**Projection metrics**

| Metric | Mathematical definition | First: $t=0$ | Last: $t=0.99$ |
| :--- | :--- | ---: | ---: |
| Relative projection residual | $r_k=\frac{\lVert\mathbf{u}_k-\mathbf{g}_k\rVert_2}{\max(\lVert\mathbf{g}_k\rVert_2,10^{-30})}$ | $3.878\times 10^{-2}$ | $6.591\times 10^{-1}$ |
| Tangent velocity (RMS) | $v_k^u=\frac{\lVert\mathbf{u}_k\rVert_2}{\sqrt{N}}$ | $6.458\times 10^{-1}$ | $7.455\times 10^{-2}$ |
| Tangent coefficient norm | $a_k=\lVert\alpha_k\rVert_2$ | $1.495\times 10^{2}$ | $8.391\times 10^{1}$ |
| Jacobian time (seconds) | $\Delta\tau_k^J=\tau_{k,\mathrm{solve}}-\tau_{k,\mathrm{basis}}$ | $4.906\times 10^{-1}$ | $5.734\times 10^{-3}$ |
| SVD solve time (seconds) | $\Delta\tau_k^{\mathrm{solve}}=\tau_{k,\mathrm{update}}-\tau_{k,\mathrm{solve}}$ | $5.397\times 10^{-3}$ | $3.203\times 10^{-3}$ |
| Map update time (seconds) | $\Delta\tau_k^X=\tau_{k,\mathrm{end}}-\tau_{k,\mathrm{update}}$ | $9.660\times 10^{-4}$ | $3.346\times 10^{-5}$ |

The median follows `torch.median` (the lower middle value for even $N$);
$Q_{0.9}$ uses linear interpolation. Projection values describe the solve at
its input time $t_k$, so the last solve is at $T-h$, while the last state is at $T$.
The wall-clock timestamps $\tau$ mark the start of each named operation and the
end of the map update. Full time histories remain in the diagnostic CSV files.
