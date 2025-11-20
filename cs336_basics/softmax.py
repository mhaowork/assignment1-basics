import torch
from torch import Tensor

def softmax(x: Tensor, dim_idx: int):
  # ... 1 ...
  max_val = x.max(dim=dim_idx, keepdim=True)[0]

  # ... dim_idx_num ...
  exp_val = torch.exp(x - max_val)

  # ... 1 ...
  sum_exp = exp_val.sum(dim=dim_idx, keepdim=True)
  return exp_val / sum_exp