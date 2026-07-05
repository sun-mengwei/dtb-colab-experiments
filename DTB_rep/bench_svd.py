import time
import torch

torch.manual_seed(0)

N, m = 20_000, 6000
dev = torch.device("cuda")

J = torch.randn(N, m, device=dev, dtype=torch.float32)
g = torch.randn(N, device=dev, dtype=torch.float32)

# Method 1: SVD on GPU fp32.
t0 = time.time()
U, S, Vh = torch.linalg.svd(J, full_matrices=False)
alpha = Vh.T @ ((U.T @ g) / S)
torch.cuda.synchronize()
print(f"GPU SVD fp32 (N={N}, m={m}): {time.time()-t0:.2f}s")

# Method 2: lstsq on GPU fp32.
t0 = time.time()
sol = torch.linalg.lstsq(J, g.unsqueeze(1))
torch.cuda.synchronize()
print(f"GPU lstsq fp32: {time.time()-t0:.2f}s, residual shape {sol.solution.shape}")

# Method 3: normal equation, ridge.
t0 = time.time()
JtJ = J.T @ J
Jtg = J.T @ g
eye = torch.eye(m, device=dev, dtype=torch.float32) * 1e-6
alpha = torch.linalg.solve(JtJ + eye, Jtg)
torch.cuda.synchronize()
print(f"GPU normal-equation ridge fp32: {time.time()-t0:.2f}s")

# Method 4: SVD on CPU fp64.
J_c = J.cpu().double()
g_c = g.cpu().double()
t0 = time.time()
U, S, Vh = torch.linalg.svd(J_c, full_matrices=False)
alpha = Vh.T @ ((U.T @ g_c) / S)
print(f"CPU SVD fp64: {time.time()-t0:.2f}s")
