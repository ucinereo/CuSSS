import torch
from cusss import SSSLU, xSSSLU
from generic_benchmark import benchmark_on_cuda

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

"""
Benchmark our SSSLU and xSSSLU implementations against the PyTorch Sigmoid-module derived SSSLU and xSSSLU function (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    @jit_fuser
    def ssslu(x, a):
        return x * (0.5 * torch.nn.functional.softsign(x) + 0.5)

    @jit_fuser
    def xssslu(x, a):
        return x * (a * torch.nn.functional.softsign(x) + 0.5)


    class SSSLUMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, a):
            return ssslu(x, a)
        
    class xSSSLUMegatron(MegatronModule):
        def __init__(self, config=None):
            super().__init__(config=config)

        def forward(self, x, a):
            return xssslu(x, a)
        
    class ReLU(torch.nn.Module):
        """Baseline: ReLU"""
        def forward(self, x, a):
            return torch.relu(x)
    
    class GELU(torch.nn.Module):
        """Baseline: GELU"""
        def forward(self, x, a):
            return torch.nn.functional.gelu(x)
        
    class SiLU(torch.nn.Module):
        """Baseline: SiLU"""
        def forward(self, x, a):
            return torch.nn.functional.silu(x)
    
    ssslu_megatron = SSSLUMegatron()
    xssslu_megatron = xSSSLUMegatron()
    ssslu_cuda = SSSLU()
    xssslu_cuda = xSSSLU()
    relu = ReLU()
    gelu = GELU()
    silu = SiLU()
    
    

    benchmark_on_cuda(modules={"xSSSLU Megatron": ssslu_megatron, "xSSSLU Cuda": ssslu_cuda, "xSSSLU Megatron": xssslu_megatron, "xSSSLU Cuda": xssslu_cuda, "ReLU": relu, "GELU": gelu, "SiLU": silu}, baseline=("xSSSLU Megatron", xssslu_megatron), mode="xSSS", out_filename="ssslu_xssslu")