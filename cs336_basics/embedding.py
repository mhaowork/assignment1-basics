import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: [torch.device | None] = None,
        dtype: [torch.dtype | None] = None,
    ):
      super().__init__()

      """
      TODO: think if this alternative is better
      w (emb_dim, vocab_size)

      x (num_token_ids, vocab_size)
      x (dot) W.T -> (num_token_ids, )


      [0, 1, 0, 0]   
      [0, 0, 0, 1]

      w.T: (vocab_size=4, emb_dim=5)
      [1,2,3,4,5]
      [6,7,1,2,3]
      [9,1,3,4,1]
      [5,4,5,1,4]

      x (dot) W.T
      [6,7,1,2,3]
      [5,4,5,1,4]
      """

      self.W = nn.Parameter(torch.empty(vocab_size, embedding_dim))

      nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
      return self.W[token_ids]



