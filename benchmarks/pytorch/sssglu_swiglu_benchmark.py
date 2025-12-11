import torch
from cusss import SSSGLU
from generic_benchmark import benchmark_on_cuda

from fused_bias_swiglu import bias_swiglu_impl

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

"""
Benchmark our xSSS implementations against the PyTorch Sigmoid-module derived xSSS function (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    @jit_fuser
    def sssglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (0.5 * (torch.nn.functional.softsign(x) + 1)) * y * x


    class SSSGLUMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, y):
            return sssglu(x, y)
    
    
    class SWIGLUMegatron(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            input = torch.cat([x, y], dim=-1)
            return bias_swiglu_impl(input, None)


    sssglu_megatron = SSSGLUMegatron()
    sssglu_cuda = SSSGLU()
    swiglu = SWIGLUMegatron()
    

    benchmark_on_cuda(modules={"SSSGLU Megatron": sssglu_megatron, "SSSGLU Cuda Naive": sssglu_cuda}, baseline=("SWIGLUMegatron", swiglu), mode="SSSGLU", out_filename="sssglu_swiglu")