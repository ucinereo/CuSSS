import torch
import torch.nn.functional as F

# xSSS

class xSSS(torch.nn.Module):
    """xSSS torch implementation"""

    class xSSSFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            # Use the custom operation
            ctx.save_for_backward(x, a)

            return xsss_forward(x, a)

        @staticmethod
        def backward(ctx, grad_output):
            # Use the custom backward operation
            x = ctx.saved_tensors[0]
            a = ctx.saved_tensors[1]

            grad_x, grad_a = xsss_backward(
                x, a, grad_output
            )

            # Return gradients in the same order as forward inputs
            return grad_x, grad_a

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.xSSSFunction.apply(x, a)

    def forward_inference(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.xSSSFunction.apply(x, a)

# Custom ops for torch.script compatibility.
@torch.library.custom_op("xsss::xsss_forward", mutates_args=())
def xsss_forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Custom SSS forward operation compatible with torch.compile"""
    return torch.ops.xsss.forward_impl(x, a)

@xsss_forward.register_fake
def _(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("xsss::xsss_backward", mutates_args=())
def xsss_backward(x: torch.Tensor, a: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom SSS backward operation compatible with torch.compile"""
    grad_x, grad_a = torch.ops.xsss.backward_impl(x, a, grad_output)
    return grad_x, grad_a


@xsss_backward.register_fake
def _(x: torch.Tensor, a: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = torch.empty_like(x)
    grad_a = torch.empty_like(a)
    return (grad_x, grad_a)