import torch
from megatron.core.jit import jit_fuser

# sss
def sss_ref_forward(x):
    return 0.5 * (x / (1.0 + x.abs()) + 1.0)
def sss_ref_backward(x):
    return 0.5 / (1.0 + x.abs()).pow(2)

@jit_fuser
def sss(x):
    return 0.5 * (torch.nn.functional.softsign(x) + 1)

# xsss
def xsss_ref_forward(x, a):
    return a * (x / (1.0 + x.abs())) + 0.5
def xsss_ref_backward(x, a):
    grad_x = a / (1 + x.abs()).pow(2)
    grad_a = (x / (1 + x.abs())).sum().view_as(a)
    return grad_x, grad_a

@jit_fuser
def xsss(x, a):
    return a * torch.nn.functional.softsign(x) + 0.5

# *** for new kernel add reference math function and pytorch jit version ***
