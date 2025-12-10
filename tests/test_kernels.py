"""
Unified parameterized tests for all CUDA kernels.

Usage:
    pytest tests/test_kernels.py                    # Run all kernel tests
    pytest tests/test_kernels.py --kernel sss       # Run only SSS tests
    pytest tests/test_kernels.py --kernel xsss      # Run only xSSS tests
    pytest tests/test_kernels.py -k forward         # Run only forward tests
    pytest tests/test_kernels.py -k backward        # Run only backward tests
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kernels import KERNEL_REGISTRY, get_kernel


@pytest.fixture
def device():
    return torch.device("cuda")


SHAPES = [(32, 32), (128, 256), (1, 1024), (1024, 1), (64, 512)]

LOSS_FUNCTIONS = [
    ("sum", lambda x: x.sum()),
    ("mean", lambda x: x.mean()),
    ("l2", lambda x: (x**2).sum()),
]


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("loss_name,loss_fn", LOSS_FUNCTIONS, ids=["sum", "mean", "l2"])
@pytest.mark.parametrize("seed", [42, 123, 456], ids=["42", "123", "456"])
class TestKernelParity:
    """Parity tests between CUDA kernels and PyTorch autograd."""

    def test_forward_parity(self, kernel_name, shape, loss_name, loss_fn, device, seed):
        """Test that CUDA forward matches PyTorch reference."""
        spec = get_kernel(kernel_name)
        torch.manual_seed(seed)

        inputs = spec.input_generator(device)
        inputs["x"] = torch.randn(*shape, device=device, requires_grad=True)

        model = spec.cuda_module().to(device)
        cuda_out = model(**inputs)

        ref_inputs = {k: v.detach().clone() for k, v in inputs.items()}
        ref_out = spec.pytorch_forward(**ref_inputs)

        torch.testing.assert_close(
            cuda_out,
            ref_out,
            msg=f"{kernel_name} forward mismatch at shape {shape} (abs_diff: {(cuda_out - ref_out).mean().item()}, rel_diff: {(cuda_out - ref_out).abs().mean() / ref_out.abs().mean()})",
        )

    def test_backward_parity(
        self, kernel_name, shape, loss_name, loss_fn, device, seed
    ):
        """Test CUDA backward against PyTorch autograd."""
        spec = get_kernel(kernel_name)
        torch.manual_seed(seed)

        # Generate input
        inputs = spec.input_generator(device)
        inputs["x"] = torch.randn(*shape, device=device)

        cuda_inputs = {
            k: v.detach().clone().requires_grad_(True) for k, v in inputs.items()
        }
        model = spec.cuda_module().to(device)
        cuda_out = model(**cuda_inputs)
        loss_fn(cuda_out).backward()
        cuda_grads = {k: v.grad.clone() for k, v in cuda_inputs.items()}

        ref_inputs = {
            k: v.detach().clone().requires_grad_(True) for k, v in inputs.items()
        }
        ref_out = spec.pytorch_forward(**ref_inputs)
        loss_fn(ref_out).backward()
        ref_grads = {k: v.grad.clone() for k, v in ref_inputs.items()}

        for key in cuda_grads:
            torch.testing.assert_close(
                cuda_grads[key],
                ref_grads[key],
                atol=1e-4,
                rtol=1e-4,
                msg=f"{kernel_name} backward grad[{key}] mismatch at shape {shape} with {loss_name} loss (abs_diff: {(cuda_grads[key] - ref_grads[key]).mean().item()}, rel_diff: {(cuda_grads[key] - ref_grads[key]).abs().mean() / ref_grads[key].abs().mean()})",
            )
