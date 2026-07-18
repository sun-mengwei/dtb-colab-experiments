# Three-player Neural--DTB result (`N=512`)

This directory contains a fresh computation of the supplied three-player
Cournot game using the Neural--DTB algorithm. The run uses uniform initial
particles on `[0,1]^3`, `sigma_i=0.1`, `h=0.005`, 200 steps, and seed 0.

Run it again from the project root with:

```bash
python replicate_three_player_game.py \
  --particles 512 \
  --device auto \
  --skip-sde-baseline \
  --output-dir outputs/three_player_dtb_n512
```

All stored algorithm arrays are finite. At `t=1`, the sample mean is
approximately `(0.314369, 0.322855, 0.317763)`.
