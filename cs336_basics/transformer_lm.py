from jaxtyping import Float
from torch import nn, Tensor

from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
    ):
        super().__init__()
        # TODO: determine dtype & device for Transformer LM
        self.input_embedding = Embedding(
            vocab_size=vocab_size, embedding_dim=d_model, device=None, dtype=None
        )
        self.transformer_blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                theta=rope_theta,
            )
            for idx in range(num_layers)
        ]
        self.output_embedding = Linear(d_in=d_model, d_out=vocab_size)
        self.rmsnorm_final = RMSNorm(d_model=d_model)

    def forward(self, in_indicies: Tensor):
        x = self.input_embedding(in_indicies)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.rmsnorm_final(x)

        output: Float[Tensor, "batch_size seq_len vocab_size"] = self.output_embedding(x)

        return output


