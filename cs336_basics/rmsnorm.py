import torch
from torch import nn

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    self.eps = eps
    


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # (batch_size, sequence_length, d_model)

    original_dtype = x.dtype

    x = x.to(dtype=torch.float32)

    mean = torch.mean(x ** 2, dim=-1, keepdim=True)

    RMS_a = torch.sqrt(mean + self.eps)

    result = x / RMS_a * self.gain

    return result.to(dtype=original_dtype)
