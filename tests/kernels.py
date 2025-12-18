"""
Kernel registry with reference PyTorch implementations for parity testing.
"""

import torch
from dataclasses import dataclass
from typing import Callable, Dict

from cusss import SSS, xSSS, SSSGLU, SSSLU, xSSSLU


@dataclass
class KernelSpec:
    """Specification for a kernel under test."""

    name: str
    cuda_module: torch.nn.Module
    pytorch_forward: Callable[..., torch.Tensor]
    input_generator: Callable[[torch.device], Dict[str, torch.Tensor]]


# SSS Kernel


def sss_pytorch_forward(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.nn.functional.softsign(x) + 0.5


def sss_input_generator(shape: tuple, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {"x": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)}


SSS_SPEC = KernelSpec(
    name="sss",
    cuda_module=SSS,
    pytorch_forward=sss_pytorch_forward,
    input_generator=sss_input_generator,
)


# xSSS Kernel


def xsss_pytorch_forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return a * torch.nn.functional.softsign(x) + 0.5


def xsss_input_generator(shape: tuple, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True),
        "a": torch.randn(1, device=device, dtype=dtype, requires_grad=True),
    }


XSSS_SPEC = KernelSpec(
    name="xsss",
    cuda_module=xSSS,
    pytorch_forward=xsss_pytorch_forward,
    input_generator=xsss_input_generator,
)

# SSSGLU Kernel

def sssglu_pytorch_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return sss_pytorch_forward(x) * y * x

def sssglu_input_generator(shape: tuple, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True),
        "y": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True),
    }

SSSGLU_SPEC = KernelSpec(
    name="sssglu",
    cuda_module=SSSGLU,
    pytorch_forward=sssglu_pytorch_forward,
    input_generator=sssglu_input_generator,
)

# SSSLU Kernel

def ssslu_pytorch_forward(x: torch.Tensor) -> torch.Tensor:
    return sss_pytorch_forward(x) * x

def ssslu_input_generator(shape: tuple, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True),
    }

SSSLU_SPEC = KernelSpec(
    name="ssslu",
    cuda_module=SSSLU,
    pytorch_forward=ssslu_pytorch_forward,
    input_generator=ssslu_input_generator,
)

# xSSSLU Kernel
def xssslu_pytorch_forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return xsss_pytorch_forward(x, a) * x

def xssslu_input_generator(shape: tuple, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.randn(*shape, device=device, dtype=dtype, requires_grad=True),
        "a": torch.randn(1, device=device, dtype=dtype, requires_grad=True),
    }

XSSSLU_SPEC = KernelSpec(
    name="xssslu",
    cuda_module=xSSSLU,
    pytorch_forward=xssslu_pytorch_forward,
    input_generator=xssslu_input_generator,
)

# ------- Kernel Registry -------

KERNEL_REGISTRY: Dict[str, KernelSpec] = {
    "sss": SSS_SPEC,
    "xsss": XSSS_SPEC,
    "sssglu": SSSGLU_SPEC,
    "ssslu": SSSLU_SPEC,
    "xssslu": XSSSLU_SPEC,
}


def get_kernel(name: str) -> KernelSpec:
    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {name}. Available: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[name]
