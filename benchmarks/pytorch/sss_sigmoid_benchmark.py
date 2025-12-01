import torch
from cusss.ops.sss_wrappers import SSS, SSS_f4
from generic_benchmark import benchmark_on_cuda, FuncType

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

"""
Benchmark our SSS implementations against the PyTorch Sigmoid-module (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    @jit_fuser
    def sss(x):
        return 0.5 * (torch.nn.functional.softsign(x) + 1)

    class SSSMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x):
            return sss(x)

    sss_megatron = SSSMegatron()
    sss_cuda = SSS()
    sss_cuda_f4 = SSS_f4()
    sigmoid = torch.nn.Sigmoid()
    relu = torch.nn.ReLU()

    benchmark_on_cuda(
        modules={
            "SSS Megatron": (sss_megatron, FuncType.DEFAULT),
            "SSS Cuda Naive": (sss_cuda, FuncType.DEFAULT),
            "SSS Cuda float4": (sss_cuda_f4, FuncType.DEFAULT),
            "ReLU torch": (relu, FuncType.DEFAULT),
        },
        baseline=("Sigmoid torch", (sigmoid, FuncType.DEFAULT)),
    )
