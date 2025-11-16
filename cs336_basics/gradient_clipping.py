import math
from typing import Iterable
import torch

def do_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
  total_sum = 0
  for p in parameters:
    if p.grad is None:
      continue
    total_sum += (p.grad * p.grad).sum().item()

  l2_norm = math.sqrt(total_sum)
    
  if l2_norm < max_l2_norm:
    return

  for p in parameters:
    if p.grad is not None:
      p.grad *= max_l2_norm / (l2_norm + eps)

