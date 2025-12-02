import pytest
import torch

from cusss.ops.sss_wrappers import SSS, SSS_f4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_forward(dtype):
    """Compare CUDA forward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=dtype, requires_grad=True)

    sss = SSS().to(device).to(dtype)
    sss_f4 = SSS_f4().to(device).to(dtype)

    expected = (0.5 * (x.float() / (1.0 + x.abs().float()) + 1.0)).to(dtype)

    # Test standard kernel
    output = sss(x)
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-5)

    # Test float4 kernel
    output_f4 = sss_f4(x)
    torch.testing.assert_close(output_f4, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_backward(dtype):
    """Compare CUDA backward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(64, 512, device=device, dtype=dtype, requires_grad=True)

    sss = SSS().to(device).to(dtype)
    sss_f4 = SSS_f4().to(device).to(dtype)

    grad_ref = 0.5 / (1.0 + x.detach().abs()).pow(2)

    # Test standard kernel
    output = sss(x)
    loss = output.sum()
    loss.backward()
    grad_cuda = x.grad.clone()
    torch.testing.assert_close(grad_cuda, grad_ref, rtol=1e-3, atol=1e-5)

    # Test float4 kernel
    x.grad = None
    output_f4 = sss_f4(x)
    loss_f4 = output_f4.sum()
    loss_f4.backward()
    grad_cuda_f4 = x.grad.clone()
    torch.testing.assert_close(grad_cuda_f4, grad_ref, rtol=1e-3, atol=1e-5)