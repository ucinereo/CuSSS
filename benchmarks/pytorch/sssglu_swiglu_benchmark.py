import torch
from cusss.ops.sss_glu_wrappers import SSSGLU
from generic_benchmark import benchmark_on_cuda, FuncType

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

"""
Benchmark our SSS implementations against the PyTorch Sigmoid-module (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    @jit_fuser
    def sssglu(x, y):
        return (0.5 * (torch.nn.functional.softsign(x) + 1)) * x * y

    class SSSGLUMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, y):
            return sssglu(x, y)
        
    @jit_fuser
    def swiglu(x, y):
        return torch.nn.functional.silu(y) * x
    
    class SwiGLUMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, y):
            return swiglu(x, y)

    sssglu_megatron = SSSGLUMegatron()
    swiglu_megatron = SwiGLUMegatron()
    sssglu_cuda = SSSGLU()

    benchmark_on_cuda(
        modules={
            "SSSGLU Megatron": (sssglu_megatron, FuncType.GLU),
            "SSSGLU CUDA": (sssglu_cuda, FuncType.GLU),
        },
        baseline=("SwiGLU Megatron", (swiglu_megatron, FuncType.GLU)),
    )
