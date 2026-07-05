"""Check Laplacian magnitude of f_theta at each snapshot."""
import os
import torch
from network import PeriodicMMNN
from dtb import write_flat_into_model, device, _laplacian_one, flat_params
from torch.func import vmap
from problem import initial_condition

dev = device()
run = "snapshots/run_diag"
meta = torch.load(os.path.join(run, "meta.pt"), weights_only=False)
a = meta["args"]
structure = meta["structure"]
frozen = meta["frozen"]

model = PeriodicMMNN(
    dim_in=5, k_per_dim=a["k_per_dim"],
    width=a["width"], rank=a["rank"], depth=a["depth"],
    activation=a["activation"], dtype=torch.float32,
).to(dev)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name in frozen:
            p.copy_(frozen[name].to(dev))

torch.manual_seed(0)
z = (torch.rand(2000, 5, device=dev) * 2.0 - 1.0)

for fn in sorted(os.listdir(run)):
    if not fn.startswith("theta_t"):
        continue
    flat = torch.load(os.path.join(run, fn), weights_only=False).to(dev)
    if torch.isnan(flat).any():
        print(f"{fn}: NaN")
        continue
    write_flat_into_model(model, flat, structure)
    theta_flat, _, _ = flat_params(model)
    theta_flat = theta_flat.to(dev)
    lap_fn = lambda x_single: _laplacian_one(theta_flat, x_single, model, structure)
    with torch.no_grad():
        u = model(z)
        lap = vmap(lap_fn)(z)
    print(f"{fn}: u in [{u.min():+.3f}, {u.max():+.3f}],"
          f" |lap| max={lap.abs().max().item():.3e},"
          f" |lap| rms={lap.pow(2).mean().sqrt().item():.3e}")
