import torch
import torch.cuda.nvtx as nvtx

from cusss import SSS

# 1. SETUP
device = torch.device("cuda")
# Create input and a pre-allocated gradient tensor to avoid scalar broadcasting
x = torch.randn(64, 1_000_000, device=device, requires_grad=True)
grad_output = torch.randn_like(x, device=device, requires_grad=True)

sss = SSS().to(device)
sigmoid = torch.nn.Sigmoid().to(device)

# 2. WARMUP (Updated to include Backward)
print("Warming up...")
for _ in range(20):
    # Native Warmup
    out = sigmoid(x)
    out.backward(grad_output)
    x.grad = None  # Reset grad to prevent unlimited accumulation

    # Custom Warmup
    out = sss(x)
    out.backward(grad_output)
    x.grad = None

torch.cuda.synchronize()

# 3. START TRACE
print("Starting trace...")
torch.cuda.profiler.start()

# --- REGION A: PyTorch Native ---
nvtx.range_push("A: PyTorch Native")
output_native = sigmoid(x)
# Pass the pre-allocated gradient directly
output_native.backward(grad_output)
torch.cuda.synchronize()
x.grad = None  # Clean up
del output_native
nvtx.range_pop()

# --- REGION B: Custom Kernel ---
nvtx.range_push("B: Custom Kernel")
output_custom = sss(x)
# Pass the pre-allocated gradient directly
output_custom.backward(grad_output)
torch.cuda.synchronize()
x.grad = None  # Clean up
del output_custom
nvtx.range_pop()

# 4. STOP TRACE
torch.cuda.profiler.stop()
print("Done.")
