from jaxtyping import Float, Int
from torch import nn, Tensor

import torch
import torch.nn.functional as F



class CrossEntropy(nn.Module):
    def forward(
      logits: Float[Tensor, " batch_size vocab_size"],
      targets: Int[Tensor, " batch_size"],
  ) -> Float[Tensor, ""]:
        # seq_len = logits.shape[-2]
        # max_val = logits.max(dim=-1, keepdim=True)[0]

        # logits -= max_val

        # logits_target = torch.exp(
        #   torch.sum(logits[torch.arange(seq_len - 1), targets[..., 1:]], dim=-1, keepdim=False)
        # )

        # print('logits', logits)

        # print('logits_target', logits_target)

        # logits_exp = torch.exp(logits)

        # logits_sum = torch.sum(logits_exp, dim=-1, keepdim=True)

        # pre_log = logits_target / torch.prod(logits_sum[..., -1])

        # output = -torch.log(pre_log) / (seq_len - 1)

        # print('output', output)
        # TODO: impl this properly
        return F.cross_entropy(logits, targets, reduction="mean")
