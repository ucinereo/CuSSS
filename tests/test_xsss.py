import pytest
import torch

from cusss import xSSS


@pytest.fixture
def xsss_setup():
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")

    x = torch.randn(64, 512, device=device, requires_grad=True)
    a = torch.randn((), device=device, requires_grad=True)
    xsss = xSSS().to(device)

    return {
        "x": x,
        "a": a,
        "xsss": xsss,
    }

def test_forward(xsss_setup):
    """Compare CUDA forward output to PyTorch implementation"""
    input = xsss_setup["x"]
    a = xsss_setup["a"]
    model = xsss_setup["xsss"]
    output = model(input, a)
    torch.testing.assert_close(output, a * input / (1.0 + input.abs()) + 0.5)


def test_backward(xsss_setup):
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
    grad_a_ref = (input.detach() * inv).sum()


    torch.testing.assert_close(x_grad_cuda, grad_x_ref)
    torch.testing.assert_close(a_grad_cuda, grad_a_ref)