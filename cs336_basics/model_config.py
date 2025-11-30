from dataclasses import dataclass

from cs336_basics.letstokenize import get_vocab_size


@dataclass
class ModelConfig:
  vocab_size: int = get_vocab_size()
  batch_size: int = 32
  context_length: int = 384
  d_model: int = 1152
  num_layers: int = 12
  num_heads: int = 18
  d_ff: int = 4608
  rope_theta: float = 10000
  lr_max: float = 3e-4
  lr_min: float = 3e-5
  warm_up_steps: int = 100
  annealing_steps: int = 3000
  gradient_accumulation_steps: int = 4
