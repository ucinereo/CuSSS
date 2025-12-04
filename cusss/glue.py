import torch

class SSS(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sss.forward(x)   

class xSSS(torch.nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.ops.xsss.forward(x, a)