import pytest
import torch

from cusss.ops.sss_glu_wrappers import SSSGLU


INPUT_SIZES = [
    (32, 256),
    (64, 512),
    (128, 1024),
    (256, 2048),
]


def sss_glu_forward_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x / (1.0 + x.abs()) * 0.5) + 0.5) * x * y


@pytest.mark.parametrize("input_size", INPUT_SIZES)
def test_forward(input_size):
    """Compare CUDA forward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(*input_size, device=device, requires_grad=True)
    y = torch.randn(*input_size, device=device, requires_grad=True)
    model = SSSGLU().to(device)

    output = model(x, y)

    torch.testing.assert_close(output, sss_glu_forward_reference(x, y))


@pytest.mark.parametrize("input_size", INPUT_SIZES)
def test_backward(input_size):
    """Compare CUDA backward output to PyTorch implementation"""
    device = torch.device("cuda")
    x = torch.randn(*input_size, device=device, requires_grad=True)
    y = torch.randn(*input_size, device=device, requires_grad=True)
    model = SSSGLU().to(device)

    output = model(x, y)

    # Simple toy loss
    loss = output.sum()
    loss.backward()

    grad_x_cuda = x.grad.clone()
    grad_y_cuda = y.grad.clone()

    # Reference implementation
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    output_ref = sss_glu_forward_reference(x_ref, y_ref)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(grad_x_cuda, x_ref.grad)
    torch.testing.assert_close(grad_y_cuda, y_ref.grad)
