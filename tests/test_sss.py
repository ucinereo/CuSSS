import pytest
import torch

from cusss import SSS


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16]
)
def test_forward(dtype):
    """Compare CUDA forward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=float, requires_grad=True).to(dtype)

    sss = SSS().to(device).to(dtype)

    expected = 0.5 * (x.detach().float() / (1.0 + x.detach().float().abs()) + 1.0)
    expected = expected.to(dtype)

    # Test standard kernel
    output = sss(x)
    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward(dtype):
    """Compare CUDA backward output to PyTorch implementation with non-uniform gradients"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(64, 512, device=device, dtype=dtype)
    sss = SSS().to(device).to(dtype)

    # Expected gradient: d(SSS)/dx * grad_output
    # where d(SSS)/dx = 0.5 / (1 + |x|)^2
    # Note that we do the order of operations as in the CUDA kernel to avoid
    # discrepancies due to rounding
    inv = 1.0 / (1.0 + x.detach().float().abs())
    grad_ref = grad_output.float() * 0.5 * inv * inv
    grad_ref = grad_ref.to(dtype)

    # Test standard kernel
    output = sss(x)
    output.backward(grad_output)
    grad_cuda = x.grad.clone()
    torch.testing.assert_close(grad_cuda.float(), grad_ref.float(), rtol=1e-3, atol=1e-5)
