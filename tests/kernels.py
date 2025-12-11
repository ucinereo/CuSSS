"""
Kernel registry with reference PyTorch implementations for parity testing.
"""

import torch
from dataclasses import dataclass
from typing import Callable, Dict

from cusss import SSS, xSSS


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


def sss_input_generator(device: torch.device) -> Dict[str, torch.Tensor]:
    return {"x": torch.randn(64, 512, device=device, requires_grad=True)}


SSS_SPEC = KernelSpec(
    name="sss",
    cuda_module=SSS,
    pytorch_forward=sss_pytorch_forward,
    input_generator=sss_input_generator,
)


# xSSS Kernel


def xsss_pytorch_forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return a * torch.nn.functional.softsign(x) + 0.5


def xsss_input_generator(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.randn(64, 64, device=device, requires_grad=True),
        "a": torch.randn(1, device=device, requires_grad=True),
    }


XSSS_SPEC = KernelSpec(
    name="xsss",
    cuda_module=xSSS,
    pytorch_forward=xsss_pytorch_forward,
    input_generator=xsss_input_generator,
)


KERNEL_REGISTRY: Dict[str, KernelSpec] = {
    "sss": SSS_SPEC,
    "xsss": XSSS_SPEC,
}


def get_kernel(name: str) -> KernelSpec:
    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {name}. Available: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[name]
