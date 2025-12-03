import torch
import torch.cuda.nvtx as nvtx

from cusss import SSS

# 1. SETUP
device = torch.device("cuda")
x = torch.randn(64, 1_000_000, device=device)
sss = SSS().to(device)
sigmoid = torch.nn.Sigmoid().to(device)

print("Warming up...")
for _ in range(20):
    _ = sss(x)
    _ = sigmoid(x)
torch.cuda.synchronize()

print("Starting trace...")
torch.cuda.profiler.start()

# --- REGION A: PyTorch Native ---
nvtx.range_push("A: PyTorch Native")
output_native = sigmoid(x)
torch.cuda.synchronize() # Wait for it to finish before popping the marker
del output_native
nvtx.range_pop()

# --- REGION B: Custom Kernel ---
sss = SSS().to(device)
nvtx.range_push("B: Custom Kernel")
# output_custom = my_custom_kernel(x, y)
output_custom = sss(x)
torch.cuda.synchronize()
nvtx.range_pop()

# 4. STOP TRACE
torch.cuda.profiler.stop()
print("Done.")