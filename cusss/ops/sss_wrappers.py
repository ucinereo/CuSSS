import torch
import torch.nn.functional as F


# SSS

class SSS(torch.nn.Module):
    """SSS torch implementation"""

    class SSSFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            # Use the custom operation
            ctx.save_for_backward(x)
            return sss_forward(x)

        @staticmethod
        def backward(ctx, grad_output):
            # Use the custom backward operation
            x = ctx.saved_tensors[0]

            grad_x = sss_backward(
                x, grad_output
            )

            # Return gradients in the same order as forward inputs
            return grad_x

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SSSFunction.apply(x)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.SSSFunction.apply(x)


# Custom ops for torch.script compatibility.
@torch.library.custom_op("sss::sss_forward", mutates_args=())
def sss_forward(x: torch.Tensor) -> torch.Tensor:
    """Custom SSS forward operation compatible with torch.compile"""
    return torch.ops.sss.forward_impl(x)


@sss_forward.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("sss::sss_backward", mutates_args=())
def sss_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Custom SSS backward operation compatible with torch.compile"""
    gradients = torch.ops.sss.backward_impl(x, grad_output)
    return gradients[0]


@sss_backward.register_fake
def _(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    grad_x = torch.empty_like(x)
    return grad_x


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

            grads = xsss_backward(
                x, a, grad_output
            )

            grad_x = grads[0]
            grad_a = grads[1]

            # Return gradients in the same order as forward inputs
            return grad_x, grad_a

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.xSSSFunction.apply(x, a)

    def forward_inference(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.xSSSFunction.apply(x, a)

# Custom ops for torch.script compatibility.
@torch.library.custom_op("sss::sss_forward", mutates_args=())
def xsss_forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Custom SSS forward operation compatible with torch.compile"""
    return torch.ops.sss.forward_impl(x, a)


@xsss_forward.register_fake
def _(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("sss::sss_backward", mutates_args=())
def xsss_backward(x: torch.Tensor, a: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Custom SSS backward operation compatible with torch.compile"""
    gradients = torch.ops.sss.backward_impl(x, a, grad_output)
    return gradients


@xsss_backward.register_fake
def _(x: torch.Tensor, a: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    grad_x = torch.empty_like(x)
    grad_a = torch.empty_like(a)
    return [grad_x, grad_a]