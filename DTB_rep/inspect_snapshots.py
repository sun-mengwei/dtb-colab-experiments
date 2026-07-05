"""Inspect snapshots to find when NaN appeared."""
import os
import torch
from network import PeriodicMMNN
from dtb import write_flat_into_model, device
from problem import initial_condition

dev = device()
run = "snapshots/run_paper"
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

# Sanity sample for evaluating each snapshot.
torch.manual_seed(0)
z = (torch.rand(4000, 5, device=dev) * 2.0 - 1.0)
ic_vals = initial_condition(z)
print(f"IC stats: min={ic_vals.min():.3f} max={ic_vals.max():.3f}"
      f" mean={ic_vals.mean():.3f}")

for fn in sorted(os.listdir(run)):
    if not fn.startswith("theta_t"):
        continue
    flat = torch.load(os.path.join(run, fn), weights_only=False).to(dev)
    nan = bool(torch.isnan(flat).any())
    write_flat_into_model(model, flat, structure)
    with torch.no_grad():
        u = model(z)
    print(f"{fn}: theta_NaN={nan}  u min={float(u.min()):.3e}"
          f"  max={float(u.max()):.3e}  rms={float(u.pow(2).mean().sqrt()):.3e}")
