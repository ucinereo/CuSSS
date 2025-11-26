import torch
import torch.nn.functional as F


import torch

class SSS(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sss.forward(x)   
    

# Same definition for float4

class SSS_f4(torch.nn.Module):
    """SSS torch implementation"""

    class SSSFunction_f4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            # Use the custom operation
            ctx.save_for_backward(x)
            return sss_forward_f4(x)

        @staticmethod
        def backward(ctx, grad_output):
            # Use the custom backward operation
            x = ctx.saved_tensors[0]

            grad_x = sss_backward_f4(
                x, grad_output
            )

            # Return gradients in the same order as forward inputs
            return grad_x

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SSSFunction_f4.apply(x)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.SSSFunction_f4.apply(x)

# Custom ops for torch.script compatibility.
@torch.library.custom_op("sss_f4::sss_forward", mutates_args=())
def sss_forward_f4(x: torch.Tensor) -> torch.Tensor:
    """Custom SSS forward operation compatible with torch.compile"""
    return torch.ops.sss.forward_impl_f4(x)


@sss_forward_f4.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("sss_f4::sss_backward", mutates_args=())
def sss_backward_f4(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Custom SSS backward operation compatible with torch.compile"""
    gradients = torch.ops.sss.backward_impl_f4(x, grad_output)
    return gradients[0]


@sss_backward_f4.register_fake
def _(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    grad_x = torch.empty_like(x)
    return grad_x