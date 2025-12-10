import torch
from cusss import SSSGLU
from generic_benchmark import benchmark_on_cuda

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

"""
Benchmark our xSSS implementations against the PyTorch Sigmoid-module derived xSSS function (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    # @jit_fuser
    # def sss(x):
    #     return 0.5 * torch.nn.functional.softsign(x) + 0.5


    # class SSSGLUMegatron(MegatronModule):
    #     def __init__(self, config=None):
    #         super().__init__(config=config)

    #     def forward(self, x, y):
    #         return sss(x) * x * y
    
    class ScaledSigmoid(torch.nn.Module):
        """Baseline: a * softsign(x) + 0.5"""
        def forward(self, x, a):
            return (1+2*a) * torch.sigmoid(x) - a



    sssglu_cuda = SSSGLU()

    def sssglu_megatron(x, y):
        return bias_swiglu_impl(torch.cat([x, y], dim=-1), None) 

    benchmark_on_cuda(modules={"SSSGLU Megatron": sssglu_megatron, "SSSGLU Cuda Naive": sssglu_cuda}, tensor_sizes=[1_000, 10_000, 100_000], baseline=("SSSGLU Megatron", sssglu_megatron), mode="SSSGLU")