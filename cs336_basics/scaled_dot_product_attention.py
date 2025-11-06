import math
from jaxtyping import Float, Bool
from torch import Tensor, nn

import torch

from cs336_basics.softmax import Softmax

class SDPA(nn.Module):

  def forward(
    self,
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
  ) -> Float[Tensor, " ... queries d_v"]:
    # Attention(Q, K, V ) = softmax(Q^T K / √dk ) V

    d_k = Q.shape[-1]
    num_queries = Q.shape[-2]
    num_keys = K.shape[-2]
    pre_softmax = Q @ K.transpose(dim0=-2, dim1=-1) / math.sqrt(d_k)
    
    if (mask is not None):
      #pre_softmax = pre_softmax + torch.zeros_like(mask, dtype=torch.float).masked_fill(~mask, float('-inf'))
      pre_softmax = pre_softmax + torch.where(mask, 0, -torch.inf)

    # "... queries keys"
    softmax = Softmax().forward(pre_softmax, dim_idx=-1)
    return softmax @ V


    



