import math
import torch


class AdamW(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, betas: tuple[float, float] = (0.9, 0.999), eps=1e-08, weight_decay=0.01):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = {"lr": lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
    super().__init__(params, defaults)

  def step(self):
    # loss = None if closure is None else closure()

    param_shape = self.param_groups[0]["params"][0].shape
    m, v = torch.zeros(param_shape), torch.zeros(param_shape)

    for group in self.param_groups:
      lr = group["lr"] # Get the learning rate.
      betas = group["betas"]
      eps = group["eps"]
      weight_decay = group["weight_decay"]

      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p]
        if len(state) == 0:
          state['t'] = 1
          state['m'] = torch.zeros_like(p)
          state['v'] = torch.zeros_like(p)
        m = state['m']
        v = state['v']
        t = state['t']

        grad = p.grad.data

        # TODO (optimize): use in-place ops
        m = betas[0] * m + (1 - betas[0]) * grad
        v = betas[1] * v + (1 - betas[1]) * grad * grad

        lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

        p.data -= lr_t * m / (torch.sqrt(v) + eps)
        p.data -= lr * weight_decay * p.data

        state["t"] = t + 1
        state["m"] = m
        state["v"] = v






        
    
