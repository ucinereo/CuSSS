import pytest
import torch

from cusss import SSS



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward(dtype):
    """Compare CUDA forward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=dtype, requires_grad=True)

    sss = SSS().to(device).to(dtype)

    expected = 0.5 * (x.float() / (1.0 + x.abs().float()) + 1.0)
    expected = expected.to(dtype)

    # Test standard kernel
    output = sss(x)
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward(dtype):
    """Compare CUDA backward output to PyTorch implementation with non-uniform gradients"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(64, 512, device=device, dtype=dtype)

    sss = SSS().to(device).to(dtype)

    # Expected gradient: d(SSS)/dx * grad_output
    # where d(SSS)/dx = 0.5 / (1 + |x|)^2
    grad_ref = (0.5 / (1.0 + x.detach().abs().float()).pow(2)) * grad_output.float()
    grad_ref = grad_ref.to(dtype)

    # Test standard kernel
    output = sss(x)
    output.backward(grad_output)
    grad_cuda = x.grad.clone()
    torch.testing.assert_close(grad_cuda, grad_ref, rtol=1e-3, atol=1e-5)