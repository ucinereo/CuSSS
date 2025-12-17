import torch
import triton
import triton.language as tl


@triton.jit
def sss_fwd_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # 0.5 * (x / (1 + |x|)) + 0.5
    numerator = x
    denominator = 1.0 + tl.abs(x)
    result = 0.5 * (numerator / denominator) + 0.5

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def sss_bwd_kernel(
    y_ptr, grad_out_ptr, grad_x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load y (output) and grad_output
    y = tl.load(y_ptr + offsets, mask=mask)
    g = tl.load(grad_out_ptr + offsets, mask=mask)

    # Math: 0.5 * grad * (1 - |2y - 1|)^2
    # This avoids division and re-reading X
    term = 1.0 - tl.abs(2.0 * y - 1.0)
    grad_x = g * 0.5 * term * term

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


def sss_triton_forward(x: torch.Tensor):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa
    sss_fwd_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def sss_triton_backward(y: torch.Tensor, grad_out: torch.Tensor):
    n_elements = y.numel()
    grad_x = torch.empty_like(y)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa
    sss_bwd_kernel[grid](y, grad_out, grad_x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return grad_x
