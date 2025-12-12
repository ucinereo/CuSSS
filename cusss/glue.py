import torch

class SSS(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sss.forward(x)   

class SSSLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.ssslu.forward(x)

class xSSS(torch.nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.ops.xsss.forward(x, a)

class xSSSLU(torch.nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.ops.xssslu.forward(x, a)
    
class SSSGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.sssglu.forward(x, y)
