from dataclasses import dataclass

from cs336_basics.letstokenize import get_vocab_size


@dataclass
class ModelConfig:
  vocab_size: int = get_vocab_size()
  batch_size: int = 32
  context_length: int = 512
  d_model: int = 1152
  num_layers: int = 12
  num_heads: int = 18
  d_ff: int = 4608
  rope_theta: float = 10000
  learning_rate: float = 3e-4
