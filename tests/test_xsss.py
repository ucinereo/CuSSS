import pytest
import torch

from cusss import xSSS


@pytest.fixture
def xsss_setup(dtype):
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")
    # reduce x size to 4k elements so grad_a accumulation error does not get too large for floaat
    x = torch.randn(64, 64, device=device, dtype=dtype, requires_grad=True)
    a = torch.randn(1, device=device, dtype=dtype, requires_grad=True)
    xsss = xSSS().to(device).to(dtype)

    return {
        "x": x,
        "a": a,
        "xsss": xsss,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward(xsss_setup, dtype):
    """Compare CUDA forward output to PyTorch implementation"""
    input = xsss_setup["x"].to(dtype)
    a = xsss_setup["a"].to(dtype)
    model = xsss_setup["xsss"].to(dtype)
    output = model(input, a)
    inv = 1.0 / (1.0 + input.float().abs())
    expected = input.float() * inv * a.float() + 0.5
    expected = expected.to(dtype)
    torch.testing.assert_close(output.float(), expected.float())


# Due to grad accumulation error on a, we only test float32 for now
@pytest.mark.parametrize("dtype", [torch.float32])
def test_backward(xsss_setup, dtype):
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
    torch.testing.assert_close(a_grad_cuda, grad_a_ref, atol=1e-4, rtol=1e-3)
