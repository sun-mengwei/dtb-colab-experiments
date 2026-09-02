# Three-player Neural--DTB: `N=2000`, depth 4

This is the requested three-player computation using four hidden
`Linear+tanh` blocks followed by the output layer. It uses the confirmed
equilibrium-consistent payoff convention

```text
r_i = sum_{j != i} x_j
Pi_i = -d - b x_i^2 + 2 b mu x_i r_i(1-r_i)
d Pi_i/dx_i = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2.
```

Configuration:

```text
N=2000
dimension=3
hidden depth=4
hidden width=32
trainable parameters=3395
selected tangent coordinates m=128
h=0.005
K=200
svd_rtol=1e-3
sigma_i=0.1
D=0.01 I
initial law=Uniform([0,1]^3)
seed=0
```

Run it again with:

```bash
python replicate_three_player_game.py \
  --particles 2000 \
  --depth 4 \
  --svd-rtol 1e-3 \
  --skip-sde-baseline \
  --device auto \
  --output-dir outputs/three_player_n2000_depth4
```

In `dtb_snapshots.png`, the four stable equilibria are red circles and the
unstable origin is a gold `X`. The run remained finite through `t=1`. Its final
mean is approximately `(0.3504, 0.3351, 0.3439)`, final relative projection
residual is `0.5112`, median distance to the nearest stable equilibrium is
`0.0904`, and `71.2%` of particles are within radius `0.15` of a stable
equilibrium.
