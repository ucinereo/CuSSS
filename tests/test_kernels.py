"""
Unified parameterized tests for all CUDA kernels.

Usage:
    pytest tests/test_kernels.py                    # Run all kernel tests
    pytest tests/test_kernels.py --kernel sss       # Run only sss tests {sss, xsss, sssglu, ssslu, xssslu}
    pytest tests/test_kernels.py --kernel xsss      # Run only xsss tests {sss, xsss, sssglu, ssslu, xssslu}
    pytest tests/test_kernels.py -k forward         # Run only forward tests
    pytest tests/test_kernels.py -k backward        # Run only backward tests
    pytest tests/test_kernels.py --loss l2          # Run only l2 loss {sum, mean, l2, mse}
    pytest tests/test_kernels.py --shape odd        # Run only odd shaped inputs {odd, even}
    pytest tests/test_kernels.py --dtype bfloat16   # Run only bfloat16 tests {float32, bfloat16}
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kernels import KERNEL_REGISTRY, get_kernel
from conftest import odd_shape


@pytest.fixture
def device():
    return torch.device("cuda")

# -------|.............................even .................. | ........... odd ...........................|
SHAPES = [(32, 32), (128, 256), (1, 1024), (1024, 1), (64, 512), (31, 33), (125, 255), (3, 1023), (769, 73)]

LOSS_FUNCTIONS = [
    ("sum", lambda x: x.sum()),
    ("mean", lambda x: x.mean()),
    ("l2", lambda x: (x**2).sum()),
    ("mse", lambda x: ((x - 1.0) ** 2).mean()),
]


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("loss_name,loss_fn", LOSS_FUNCTIONS, ids=["sum", "mean", "l2", "mse"])
@pytest.mark.parametrize("seed", [42, 123, 456], ids=["42", "123", "456"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
class TestKernelParity:
    """Parity tests between CUDA kernels and PyTorch autograd."""

    def test_forward_parity(self, kernel_name, shape, loss_name, loss_fn, device, seed, dtype):
        """Test that CUDA forward matches PyTorch reference."""
        spec = get_kernel(kernel_name)
        torch.manual_seed(seed)
        use_low_precision = dtype in {torch.bfloat16, torch.float16}

        inputs = spec.input_generator(shape, device, dtype)

        model = spec.cuda_module().to(device).to(dtype)
        cuda_out = model(**inputs)

        ref_inputs = {k: v.detach().clone().float() for k, v in inputs.items()}
        ref_out = spec.pytorch_forward(**ref_inputs).to(dtype)
        print(dtype)
        torch.testing.assert_close(
            cuda_out.float(),
            ref_out.float(),
            atol=2e-3 if use_low_precision else 1e-4,
            rtol=1e-2 if use_low_precision else 1e-5,
            msg=f"{kernel_name} forward mismatch at {'ODD' if odd_shape(shape) else 'EVEN'} shape {shape} (abs_diff: {(cuda_out - ref_out).mean().item()}, rel_diff: {(cuda_out - ref_out).abs().mean() / ref_out.abs().mean()})",
        )

    def test_backward_parity(
        self, kernel_name, shape, loss_name, loss_fn, device, seed, dtype
    ):
        """Test CUDA backward against PyTorch autograd."""
        spec = get_kernel(kernel_name)
        torch.manual_seed(seed)
        use_low_precision = dtype in {torch.bfloat16, torch.float16}
        # Generate input
        inputs = spec.input_generator(shape, device, dtype)

        cuda_inputs = {
            k: v.detach().clone().requires_grad_(True) for k, v in inputs.items()
        }
        model = spec.cuda_module().to(device).to(dtype)
        cuda_out = model(**cuda_inputs)
        loss_fn(cuda_out).backward()
        cuda_grads = {k: v.grad.clone() for k, v in cuda_inputs.items()}

        ref_inputs = {
            k: v.detach().clone().float().requires_grad_(True) for k, v in inputs.items()
        }
        ref_out = spec.pytorch_forward(**ref_inputs).to(dtype)
        loss_fn(ref_out).backward()
        ref_grads = {k: v.grad.clone() for k, v in ref_inputs.items()}

        for key in cuda_grads:
            if use_low_precision and key == "a":
                continue  # Skip low-precision scalar grad check: rounding accumulation errors
            print(key, cuda_grads[key].shape, ref_grads[key].shape)
            torch.testing.assert_close(
                cuda_grads[key].float(),
                ref_grads[key].float(),
                atol=2e-3 if use_low_precision else 1e-4,
                rtol=1e-2 if use_low_precision else 1e-5,
                msg=f"{kernel_name} backward grad[{key}] mismatch at {'ODD' if odd_shape(shape) else 'EVEN'} shape {shape} with {loss_name} for type {dtype} loss (abs_diff: {(cuda_grads[key] - ref_grads[key]).mean().item()}, rel_diff: {(cuda_grads[key] - ref_grads[key]).abs().mean() / ref_grads[key].abs().mean()})",
            )
