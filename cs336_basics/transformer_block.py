from torch import nn, Tensor
import torch

from cs336_basics.multihead_self_attention import MHSA
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RoPE
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: [torch.device | None] = None,
        dtype: [torch.dtype | None] = None,
    ):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device)
        self.rope = RoPE(
            theta=theta, d_k=(d_model // num_heads), max_seq_len=max_seq_len, device=device,
        )
        self.mhsa = MHSA(d_model=d_model, num_heads=num_heads, device=device)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device)


    def forward(self, x: Tensor) -> Tensor:
        y = x + self.mhsa.forward(
            self.rmsnorm1.forward(x), rope=self.rope, token_positions=None
        )

        z = y + self.ffn.forward(self.rmsnorm2.forward(y))
        return z
