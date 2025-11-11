from jaxtyping import Float
import torch
from torch import Tensor, nn

class SwiGLU(nn.Module):
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()

    self.d_model = d_model
    self.d_ff = d_ff

    self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
    self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
    self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

    # TODO: verify this: You should set dff to approximately 8/3 × d_model
    # in your implementation, while ensuring that the dimensionality of
    # the inner feed-forward layer is a multiple of 64 to make good use of your hardware

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    # SiLU(x) = x·σ(x) = x / (1 + e ** −x)
    # FFN(x) = SwiGLU(x, W1, W2, W3) = (SiLU(xW1) ⊙ xW3)w2

    xw1: Float[Tensor, "... d_ff"] = x @ self.w1.T
    xw3: Float[Tensor, "... d_ff"] = x @ self.w3.T
    silu: Float[Tensor, "... d_ff"] = xw1 * torch.sigmoid(xw1)


    return (silu * xw3) @ self.w2.T
    

