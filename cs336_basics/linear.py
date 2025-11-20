from math import sqrt
import torch
from torch import nn

class Linear(torch.nn.Module):
  def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
    super().__init__()

    weights = torch.empty(d_out, d_in, device=device, dtype=dtype)
    self.W = nn.Parameter(weights)

    std = sqrt(2.0 / (d_in + d_out))
    nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.W.T

    
