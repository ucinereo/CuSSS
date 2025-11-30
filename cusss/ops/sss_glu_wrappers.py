from typing import Tuple

import torch
import torch.nn.functional as F



class SSSGLU(torch.nn.Module):
    """
    SSSGLU torch implementation

    SSSGLU(x, y) = SSS(x) * x * y
                 = (x / (1.0 + x.abs()) * 0.5 + 0.5) * x * y
    """

    class SSSGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Use the custom operation
            ctx.save_for_backward(x, y)
            # TODO: Implement forward
            return sss_glu_forward(x, y)

        @staticmethod
        def backward(ctx, grad_output):
            # Use the custom backward operation
            x = ctx.saved_tensors[0]
            y = ctx.saved_tensors[1]

            grad_x, grad_y = sss_glu_backward(
                x, y, grad_output
            )

            # TODO: Implement backward

            # Return gradients in the same order as forward inputs
            return grad_x, grad_y

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.SSSGLUFunction.apply(x, y)

    def forward_inference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.SSSGLUFunction.apply(x, y)


# Custom ops for torch.script compatibility.
@torch.library.custom_op("sss_glu::sss_glu_forward", mutates_args=())
def sss_glu_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Custom SSSGLU forward operation compatible with torch.compile"""
    return torch.ops.sss_glu.forward_impl(x, y)


@sss_glu_forward.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("sss_glu::sss_glu_backward", mutates_args=())
def sss_glu_backward(x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom SSSGLU backward operation compatible with torch.compile"""
    gradients = torch.ops.sss_glu.backward_impl(x, y, grad_output)
    return gradients[0], gradients[1]


@sss_glu_backward.register_fake
def _(x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)
    return grad_x, grad_y
