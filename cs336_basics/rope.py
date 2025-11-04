import math
from jaxtyping import Float
from sympy import O
import torch
from torch import nn, Tensor


class RoPE(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super().__init__()

    self.theta = theta
    self.d_k = d_k
    self.max_seq_len = max_seq_len
    assert d_k % 2 == 0, "RoPE requires even d_k"

    #  θi,k = i / Θ ^((2k−2)/d)
    cos_vals = torch.zeros(max_seq_len, d_k // 2)
    sin_vals = torch.zeros(max_seq_len, d_k // 2)
    for i in range(max_seq_len):
      for k in range(d_k // 2):
        theta_i_k = i / theta ** (2 * k / d_k)
        sin_vals[i][k] = math.sin(theta_i_k)
        cos_vals[i][k] = math.cos(theta_i_k)

    self.register_buffer('cos_vals', cos_vals)
    self.register_buffer('sin_vals', sin_vals)

    if device is not None:
      self.to(device)

    
  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)

    # [..., seq_len, d_k // 2]
    first_elems = x_pairs[..., 0]
    second_elems = x_pairs[..., 1]

    # print(x[..., 1, :])

    # result: [..., seq_len d_k // 2]
    cos = self.cos_vals[token_positions]
    sin = self.sin_vals[token_positions]

    first_elems_new = first_elems * cos - second_elems * sin
    second_elems_new = first_elems * sin + second_elems * cos

    x_pairs[..., 0] = first_elems_new
    x_pairs[..., 1] = second_elems_new

    # for i in range(x.shape[-2]):
    #   cos_val = self.cos_vals[token_positions[i]]
    #   sin_val = self.sin_vals[token_positions[i]]

    #   first_elem = first_elems[..., i, :]
    #   second_elem = second_elems[..., i, :] 

    #   # print('old', first_elem)
    #   first_elem_new = first_elem * cos_val - second_elem * sin_val
    #   second_elem_new = first_elem * sin_val + second_elem * cos_val

    #   # print(first_elem)
    #   first_elems[..., i, :] = first_elem_new
    #   second_elems[..., i, :] = second_elem_new
    #   # print(x[..., i, :])


    # # print('newf\n', x[..., 1, :])
    return x


