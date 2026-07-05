"""Microbenchmark for dtb_basis_matrices at paper scale."""
import time
import torch
from network import PeriodicMMNN, count_trainable
from dtb import flat_params, dtb_basis_matrices, device

torch.manual_seed(0)

dev = device()
print("device:", dev)

# Paper-scale setup: width 366, rank 25, depth 7, K=40 -> PL out 200.
model = PeriodicMMNN(dim_in=5, k_per_dim=40, width=366, rank=25, depth=7,
                     activation="tanh").to(dev)
M = count_trainable(model)
print("trainable params:", M)

theta_flat, structure, _ = flat_params(model)
theta_flat = theta_flat.to(dev)

m = 6000
sel = torch.randperm(M, device=dev)[:m].sort().values

N = 20_000
x = (torch.rand(N, 5, device=dev) * 2.0 - 1.0)

# Warmup with small N.
_ = dtb_basis_matrices(theta_flat, sel, x[:200], model, structure, chunk=200)
if dev.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

t0 = time.time()
u, J, lap, LJ = dtb_basis_matrices(theta_flat, sel, x, model, structure,
                                   chunk=1000)
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0
print(f"N={N}, m={m}: dtb_basis_matrices took {dt:.2f}s")
print(f"peak GPU mem (MB):"
      f" {torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0:.1f}")
print("J  shape", J.shape, "dtype", J.dtype)
print("LJ shape", LJ.shape)
