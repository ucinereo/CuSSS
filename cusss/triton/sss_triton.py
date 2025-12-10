import torch
import torch.nn as nn
import triton
import triton.language as tl

# ===================================================================
# TRITON KERNELS


@triton.jit
def sss_forward_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Math: 0.5 * (x / (1 + |x|)) + 0.5
    numerator = x
    denominator = 1.0 + tl.abs(x)
    result = 0.5 * (numerator / denominator) + 0.5

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def sss_backward_kernel(
    y_ptr, grad_out_ptr, grad_x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)
    g = tl.load(grad_out_ptr + offsets, mask=mask)

    term = 1.0 - tl.abs(2.0 * y - 1.0)
    grad_x = g * 0.5 * term * term

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


# ===================================================================
# KERNEL LAUNCHERS


def _sss_forward(x: torch.Tensor):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa
    sss_forward_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def _sss_backward(y: torch.Tensor, grad_out: torch.Tensor):
    n_elements = y.numel()
    # Ensure contiguity for correct pointer math
    y = y.contiguous()
    grad_out = grad_out.contiguous()

    grad_x = torch.empty_like(y)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa
    sss_backward_kernel[grid](y, grad_out, grad_x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return grad_x


# ===================================================================
# AUTOGRAD FUNCTION (The Bridge)


class SSSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _sss_forward(x)

        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors

        grad_x = _sss_backward(y, grad_output)

        return grad_x


# ===================================================================
# Torch Module Wrapper


class SSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SSSFunction.apply(x)
