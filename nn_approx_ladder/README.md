# NN Approximation Ladder

Small experiments for studying how neural networks approximate functions,
then taking the first step toward PDE solving.

This folder is standalone. It does not import the existing DTB reproduction
code.

## Setup

```bash
cd /Users/msun247/Downloads/DTB_rep/nn_approx_ladder
python -m pip install -r requirements.txt
```

## 1. Small NN

Start with a tiny one-hidden-layer network on a smooth function.

```bash
python train_approx.py --targets smooth --models tiny --steps 1000
```

## 2. FCNN

Compare the tiny network with a conventional fully connected network.

```bash
python train_approx.py --targets smooth localized --models tiny fcnn
```

## 3. Single-Layer MCNN

Add one random-feature layer:

```text
h(x) = A sigma(Wx + b) + c
```

Here `W,b` are frozen after random initialization, while `A,c` are trained.

```bash
python train_approx.py --targets smooth oscillatory localized \
                       --models tiny fcnn mcnn
```

## 4. Stacked MMNN

Stack random-feature layers and compare against FCNN/MCNN.

```bash
python train_approx.py --targets smooth oscillatory localized multiscale \
                       --models tiny fcnn mcnn mmnn
```

For harder targets, increase width/rank/depth:

```bash
python train_approx.py --targets oscillatory localized multiscale piecewise \
                       --models fcnn mcnn mmnn \
                       --width 256 --rank 32 --mmnn_depth 4 \
                       --steps 5000
```

Outputs go to `runs/approx/`:

* one PNG per target
* `metrics_summary.png`, a log-scale bar chart comparing model errors
* `metrics.csv`
* `config.json`

## 4a. Notebook: Visual FCNN vs MMNN Comparison

Open the notebook when you want an interactive view of the differences between
a regular neural network and MMNN:

```bash
jupyter notebook notebooks/mmnn_vs_fcnn.ipynb
```

The notebook trains `fcnn` and `mmnn` on the harder targets:

* `oscillatory`
* `localized`
* `multiscale`
* `piecewise`

It shows target overlays, signed error curves, log-scale metric bars, and the
difference between trainable parameters and frozen random-feature buffers.

## 5. First PDE: 1-D Poisson PINN

Solve:

```text
-u''(x) = f(x),  x in (0, 1)
u(0) = u(1) = 0
```

The right-hand side is generated from a known exact solution so the error is
measurable.

```bash
python train_poisson_pinn.py --model fcnn --steps 5000
python train_poisson_pinn.py --model mmnn --steps 5000
```

Outputs go to `runs/poisson/`.

## Files

```text
models.py             TinyNN, FCNN, MCNN, MMNN from scratch
targets.py            approximation targets and Poisson exact/RHS functions
train_approx.py       function approximation training and plots
train_poisson_pinn.py first PDE/PINN experiment
notebooks/            interactive FCNN vs MMNN visual comparisons
requirements.txt      Python dependencies
```

## Good Experiments

Change one thing at a time:

* target type: `smooth`, `oscillatory`, `localized`, `multiscale`, `piecewise`
* model family: `tiny`, `fcnn`, `mcnn`, `mmnn`
* width/rank/depth
* activation: `tanh`, `gelu`, `sin`, `relu`
* training steps and sample count

For PDE experiments, prefer smooth activations such as `tanh`, `gelu`, or
`sin`; ReLU has weak second-derivative behavior for this PINN loss.
