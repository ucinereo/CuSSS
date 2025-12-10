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


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
class TestKernelParity:
    """Parity tests between CUDA kernels and PyTorch reference implementations."""

    def test_forward_parity(self, kernel_name, device):
        """Test that CUDA forward matches PyTorch reference."""
        spec = get_kernel(kernel_name)
        inputs = spec.input_generator(device)

        model = spec.cuda_module().to(device)
        cuda_out = model(**inputs)

        # Detach inputs for reference computation
        ref_inputs = {k: v.detach() for k, v in inputs.items()}
        ref_out = spec.pytorch_forward(**ref_inputs)

        torch.testing.assert_close(
            cuda_out, ref_out, msg=f"{kernel_name} forward mismatch"
        )

    def test_backward_parity(self, kernel_name, device):
        """Test that CUDA backward matches PyTorch reference."""
        spec = get_kernel(kernel_name)
        inputs = spec.input_generator(device)

        model = spec.cuda_module().to(device)
        cuda_out = model(**inputs)

        loss = cuda_out.sum()
        loss.backward()

        cuda_grads = tuple(v.grad.clone() for v in inputs.values())

        # Compute reference gradients
        ref_inputs = {k: v.detach() for k, v in inputs.items()}
        grad_out = torch.ones_like(cuda_out)
        ref_grads = spec.pytorch_backward(**ref_inputs, grad_out=grad_out)

        for i, (cuda_g, ref_g) in enumerate(zip(cuda_grads, ref_grads)):
            torch.testing.assert_close(
                cuda_g, ref_g, msg=f"{kernel_name} backward grad[{i}] mismatch"
            )

    def test_forward_deterministic(self, kernel_name, device):
        """Test that CUDA forward is deterministic."""
        spec = get_kernel(kernel_name)
        inputs = spec.input_generator(device)

        model = spec.cuda_module().to(device)

        # Detach to avoid graph issues on second call
        inputs_detached = {k: v.detach() for k, v in inputs.items()}

        out1 = model(**inputs_detached)
        out2 = model(**inputs_detached)

        torch.testing.assert_close(
            out1, out2, msg=f"{kernel_name} forward not deterministic"
        )

    def test_dtype_float32(self, kernel_name, device):
        """Test kernel with float32 dtype."""
        spec = get_kernel(kernel_name)
        inputs = spec.input_generator(device)
        inputs = {k: v.float() for k, v in inputs.items()}

        model = spec.cuda_module().to(device)
        out = model(**inputs)

        assert out.dtype == torch.float32


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
class TestKernelShapes:
    """Shape handling tests for CUDA kernels."""

    @pytest.mark.parametrize(
        "shape",
        [(32, 32), (128, 256), (1, 1024), (1024, 1), (64, 512)],
    )
    def test_various_shapes(self, kernel_name, device, shape):
        """Test kernel with various input shapes."""
        spec = get_kernel(kernel_name)

        # Generate base inputs and reshape x
        inputs = spec.input_generator(device)
        inputs["x"] = torch.randn(*shape, device=device, requires_grad=True)

        model = spec.cuda_module().to(device)
        out = model(**inputs)

        assert out.shape == shape

        # Test backward works
        out.sum().backward()
        assert inputs["x"].grad is not None
        assert inputs["x"].grad.shape == shape
