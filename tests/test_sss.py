import pytest
import torch

from cusss.ops.sss_wrappers import SSS

@pytest.fixture
def sss_setup():
    """Fixture to set up SSS instances and test input"""
    device = torch.device("cuda")
    
    x = torch.randn(64, 512, device=device, requires_grad=True)
    sss = SSS().to(device)
    # sss_f4 = SSS_f4().to(device)

    return {
        "x": x,
        "sss": sss,
        # "sss_float4": sss_f4
    }

def test_forward(sss_setup):
    """Compare CUDA forward output to PyTorch implementation"""
    input = sss_setup["x"]
    model = sss_setup["sss"]
    output = model(input)
    torch.testing.assert_close(output, 0.5 * (input / (1.0 + input.abs()) + 1.0))
    
    # Also test cuda float4 kernel
    # model_f4 = sss_setup["sss_float4"]
    # output_f4 = model_f4(input)
    # torch.testing.assert_close(output_f4, 0.5 * (input / (1.0 + input.abs()) + 1.0))

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

    # Also test cuda float4 kernel
    # input.grad = None
    # model_f4 = sss_setup["sss_float4"]

    # output_f4 = model_f4(input)

    # loss_f4 = output_f4.sum()
    # loss_f4.backward()

    # grad_cuda_f4 = input.grad.clone()
    # grad_ref = 0.5 / (1.0 + input.detach().abs()).pow(2)

    # torch.testing.assert_close(grad_cuda_f4, grad_ref)