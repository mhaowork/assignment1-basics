from dataclasses import dataclass

from cs336_basics.letstokenize import get_vocab_size


@dataclass
class ModelConfig:
  vocab_size: int = get_vocab_size()
  batch_size: int = 32 
  context_length: int = 256
  d_model: int = 384
  num_layers: int = 4
  num_heads: int = 16
  d_ff: int = 1536
  rope_theta: float = 10000

