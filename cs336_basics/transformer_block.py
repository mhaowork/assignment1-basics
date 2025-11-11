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
        weights: dict[str, Tensor],
        weight_key_prefix: str = '',
    ):
        super().__init__()

        # self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(d_model=d_model)
        self.rmsnorm1.load_state_dict({"gain": weights[weight_key_prefix + "ln1.weight"]})

        self.rope = RoPE(
            theta=theta, d_k=(d_model // num_heads), max_seq_len=max_seq_len
        )
        self.mhsa = MHSA(d_model=d_model, num_heads=num_heads)

        self.mhsa.load_state_dict(
            {
                "q_weights": weights[weight_key_prefix + "attn.q_proj.weight"],
                "k_weights": weights[weight_key_prefix + "attn.k_proj.weight"],
                "v_weights": weights[weight_key_prefix + "attn.v_proj.weight"],
                "output_weights": weights[weight_key_prefix + "attn.output_proj.weight"],
            }
        )

        self.rmsnorm2 = RMSNorm(d_model=d_model)
        self.rmsnorm2.load_state_dict({"gain": weights[weight_key_prefix + "ln2.weight"]})

        self.ffn = SwiGLU(d_model, d_ff)
        self.ffn.load_state_dict(
            {
                "w1": weights[weight_key_prefix + "ffn.w1.weight"],
                "w2": weights[weight_key_prefix + "ffn.w2.weight"],
                "w3": weights[weight_key_prefix + "ffn.w3.weight"],
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.mhsa.forward(
            self.rmsnorm1.forward(x), rope=self.rope, token_positions=None
        )

        z = y + self.ffn.forward(self.rmsnorm2.forward(y))
        return z
