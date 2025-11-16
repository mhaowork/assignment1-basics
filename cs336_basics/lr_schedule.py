import math


def get_lr_cosine_schedule(
    t: int, lr_max: float, lr_min: float, warm_up_steps: int, annealing_steps: int
):
  if t < warm_up_steps:
    return t / warm_up_steps * lr_max

  if t > annealing_steps:
    return lr_min

  lr = 0.5 * (lr_max - lr_min)

  lr *= 1 + math.cos((t - warm_up_steps) / (annealing_steps - warm_up_steps) * math.pi)

  lr += lr_min

  return lr
