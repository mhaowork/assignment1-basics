from torch import nn, Tensor

import torch
from jaxtyping import Float, Int
from einops import rearrange


from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import SDPA

import torch.nn.functional as F

class MHSA(nn.Module):
  def __init__(self, d_model, num_heads, device = None) -> None:
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads

    std = 0.02 # GPT-2 style TODO verify this
    self.q_weights = nn.Parameter(torch.randn(d_model, d_model, device=device) * std)
    self.k_weights = nn.Parameter(torch.randn(d_model, d_model, device=device) * std)
    self.v_weights = nn.Parameter(torch.randn(d_model, d_model, device=device) * std)
    self.output_weights = nn.Parameter(torch.randn(d_model, d_model, device=device) * std)

    self.sdpa = SDPA()


  def forward(
    self,
    x: Float[Tensor, " ... sequence_length d_in"],
    rope: RoPE | None = None,
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
  ) -> Float[Tensor, " ... sequence_length d_out"]:
  
    """
    WqX -> ... seq_len d_k
    """
    seq_len = x.size(-2)

    Q = x @ self.q_weights.T
    K = x @ self.k_weights.T
    V = x @ self.v_weights.T

    # print(f"Q shape: {Q.shape}")

    Q = rearrange(Q, '... L (h e) -> ... h L e', h=self.num_heads)
    K = rearrange(K, '... L (h e) -> ... h L e', h=self.num_heads)
    V = rearrange(V, '... L (h e) -> ... h L e', h=self.num_heads)

    # print(f"Q_sliced_stacked shape: {Q.shape}")

    if rope is not None:
      if token_positions is None:
        token_positions = torch.arange(seq_len)
      Q = rope.forward(Q, token_positions=token_positions)
      K = rope.forward(K, token_positions=token_positions)


    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu().T
    MHA = self.sdpa(Q, K, V, mask=mask)

    # print(f"MHA shape after SDPA: {MHA.shape}")

    MHA = rearrange(MHA, '... h L d -> ... L (h d)')
    # print(f"MHA shape after rearrange: {MHA.shape}")


    return MHA @ self.output_weights.T