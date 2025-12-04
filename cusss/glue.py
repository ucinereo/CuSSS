import torch

class SSS(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sss.forward(x)   

class SSSGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.sssglu.forward(x, y)