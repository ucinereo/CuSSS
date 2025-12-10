import pytest
import torch
from megatron.core.jit import jit_fuser

from cusss import SSS


@pytest.fixture
def sss_setup():
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")

    x = torch.randn(64, 512, device=device, requires_grad=True)
    sss = SSS().to(device)

    return {
        "x": x,
        "sss": sss,
    }

def test_forward_logic(sss_setup):
    """Compare CUDA forward output to PyTorch implementation"""
    input = sss_setup["x"]
    model = sss_setup["sss"]
    output = model(input)
    torch.testing.assert_close(output, 0.5 * (input / (1.0 + input.abs()) + 1.0))


def test_backward_logic(sss_setup):
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

def test_cuda_forward_pytorch_parity(sss_setup):
    """Ensure CUDA SSS matches PyTorch SSS implementation on forward pass"""
    input = sss_setup["x"]
    model = sss_setup["sss"]
    output_cuda = model(input)

    @jit_fuser
    def sss(x):
        return 0.5 * (torch.nn.functional.softsign(x) + 1)

    # PyTorch implementation
    output_ref = sss(input)

    torch.testing.assert_close(output_cuda, output_ref)

def test_cuda_backward_pytorch_parity(sss_setup):
    """Ensure CUDA SSS matches PyTorch SSS implementation on backward pass"""
    input = sss_setup["x"]
    model = sss_setup["sss"]
    output_cuda = model(input)

    @jit_fuser
    def sss(x):
        return 0.5 * (torch.nn.functional.softsign(x) + 1)

    # PyTorch implementation
    output_ref = sss(input)

    # Backward pass for CUDA
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    grad_cuda = input.grad.clone()

    # Reset gradients
    input.grad.zero_()

    # Backward pass for PyTorch
    loss_ref = output_ref.sum()
    loss_ref.backward()
    grad_ref = input.grad.clone()

    torch.testing.assert_close(grad_cuda, grad_ref)