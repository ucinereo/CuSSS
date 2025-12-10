import pytest
import torch
from megatron.core.jit import jit_fuser

from cusss import xSSS



@pytest.fixture
def xsss_setup():
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")

    # reduce x size to 4k elements so grad_a accumulation error does not get too large for floaat
    x = torch.randn(64, 64, device=device, requires_grad=True)
    a = torch.randn(1, device=device, requires_grad=True)
    xsss = xSSS().to(device)

    return {
        "x": x,
        "a": a,
        "xsss": xsss,
    }

def test_forward_logic(xsss_setup):
    """Compare CUDA forward output to PyTorch implementation"""
    input = xsss_setup["x"]
    a = xsss_setup["a"]
    model = xsss_setup["xsss"]
    output = model(input, a)
    torch.testing.assert_close(output, a * input / (1.0 + input.abs()) + 0.5)


def test_backward_logic(xsss_setup):
    """Compare CUDA backward output to PyTorch implementation"""
    input = xsss_setup["x"]
    a = xsss_setup["a"]
    model = xsss_setup["xsss"]

    output = model(input, a)

    # Simple toy loss
    loss = output.sum()
    loss.backward()

    x_grad_cuda = input.grad.clone()
    a_grad_cuda = a.grad.clone()

    inv = 1.0 / (1.0 + input.detach().abs())
    grad_x_ref = (inv * inv) * a.detach()
    grad_a_ref = (input.detach() * inv).sum().view_as(a)

    torch.testing.assert_close(x_grad_cuda, grad_x_ref)
    torch.testing.assert_close(a_grad_cuda, grad_a_ref)

def test_cuda_forward_pytorch_parity(xsss_setup):
    """Ensure CUDA xSSS matches PyTorch xSSS implementation on forward pass"""
    input = xsss_setup["x"]
    a = xsss_setup["a"]
    model = xsss_setup["xsss"]
    output_cuda = model(input, a)

    @jit_fuser
    def xsss(x, a):
        return a * (x / (1.0 + x.abs())) + 0.5

    # PyTorch implementation
    output_ref = xsss(input, a)

    torch.testing.assert_close(output_cuda, output_ref)

def test_cuda_backward_pytorch_parity(xsss_setup):
    """Ensure CUDA xSSS matches PyTorch xSSS implementation on backward pass"""
    input = xsss_setup["x"]
    a = xsss_setup["a"]
    model = xsss_setup["xsss"]
    output_cuda = model(input, a)

    @jit_fuser
    def xsss(x, a):
        return a * (x / (1.0 + x.abs())) + 0.5

    # PyTorch implementation
    output_ref = xsss(input, a)

    # Simple toy loss
    loss = output_ref.sum()
    loss.backward()

    grad_x_ref = input.grad.clone()
    grad_a_ref = a.grad.clone()

    # Reset gradients for CUDA backward pass
    input.grad.zero_()
    a.grad.zero_()

    loss_cuda = output_cuda.sum()
    loss_cuda.backward()

    grad_x_cuda = input.grad.clone()
    grad_a_cuda = a.grad.clone()

    torch.testing.assert_close(grad_x_cuda, grad_x_ref)
    torch.testing.assert_close(grad_a_cuda, grad_a_ref)