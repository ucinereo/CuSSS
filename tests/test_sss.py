import pytest
import torch

from cusss.ops.sss_wrappers import SSS

@pytest.fixture
def sss_setup():
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")
    
    x = torch.randn(10, 10, device=device, requires_grad=True)
    sss = SSS().to(device)

    return {
        "x": x,
        "sss": sss,
    }

def test_forward(sss_setup):
    """Compare CUDA forward output to PyTorch implementation"""
    input = sss_setup["x"]
    model = sss_setup["sss"]
    output = model(input)
    torch.testing.assert_close(output, 0.5 * (input / (1.0 + input.abs()) + 1.0))

def test_backward(sss_setup):
    """Compare CUDA backward output to PyTorch implementation"""
    input = sss_setup["x"]
    model = sss_setup["sss"]

    output = model(input)

    # Simple toy loss
    loss = output.sum()
    loss.backward()

    grad_cuda = input.grad.clone()
    grad_ref = 0.5 / (1.0 + input.detach().abs()).pow(2)

    torch.testing.assert_close(grad_cuda, grad_ref)