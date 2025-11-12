from torch import nn, Tensor

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
    ):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model=d_model)
        self.rope = RoPE(
            theta=theta, d_k=(d_model // num_heads), max_seq_len=max_seq_len
        )
        self.mhsa = MHSA(d_model=d_model, num_heads=num_heads)
        self.rmsnorm2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model, d_ff)


    def forward(self, x: Tensor) -> Tensor:
        y = x + self.mhsa.forward(
            self.rmsnorm1.forward(x), rope=self.rope, token_positions=None
        )

        z = y + self.ffn.forward(self.rmsnorm2.forward(y))
        return z
