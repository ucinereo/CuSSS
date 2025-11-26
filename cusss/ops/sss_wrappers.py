import torch


import torch

class SSS(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sss.forward(x)   
    
