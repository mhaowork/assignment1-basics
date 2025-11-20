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
          state['t'] = 0
          state['m'] = torch.zeros_like(p)
          state['v'] = torch.zeros_like(p)
        m = state['m']
        v = state['v']
        t = state['t'] + 1

        grad = p.grad.data

        m.mul_(betas[0]).add_(grad, alpha=(1 - betas[0]))
        v.mul_(betas[1]).addcmul_(grad, grad, value=(1 - betas[1]))

        lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

        p.data -= lr_t * m / (torch.sqrt(v) + eps)
        p.data -= lr * weight_decay * p.data

        state["t"] = t






        
    
