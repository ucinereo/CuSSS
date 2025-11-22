import torch
from cusss.ops.xsss_wrappers import xSSS
from generic_benchmark import benchmark_on_cuda

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

"""
Benchmark our xSSS implementations against the PyTorch Sigmoid-module derived xSSS function (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    @jit_fuser
    def xsss(x, a):
        return a * torch.nn.functional.softsign(x) + 0.5


    class xSSSMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, a):
            return xsss(x, a)
    
    class ScaledSigmoid(torch.nn.Module):
        """Baseline: a * softsign(x) + 0.5"""
        def forward(self, x, a):
            return (1+2*a) * torch.sigmoid(x) - a


    xsss_megatron = xSSSMegatron()
    xsss_cuda = xSSS()
    scaled_sigmoid = ScaledSigmoid()

    benchmark_on_cuda(modules={"xSSS Megatron": xsss_megatron, "xSSS Cuda Naive": xsss_cuda, }, tensor_sizes=[1_000, 10_000, 100_000], baseline=("Scaled Sigmoid", scaled_sigmoid), number_inputs=2)