import torch
from cusss.ops.sss_wrappers import SSS
from generic_benchmark import benchmark_on_cuda

"""
Benchmark our SSS implementations against the PyTorch Sigmoid-module (as well as ReLU as a second baseline for performant activation functions)
"""
if __name__ == "__main__":

    sss_cuda = SSS()
    sigmoid = torch.nn.Sigmoid()  
    relu = torch.nn.ReLU()  

    benchmark_on_cuda(modules={"SSS Cuda": sss_cuda, "ReLU torch": relu}, baseline=("Sigmoid torch", sigmoid))
