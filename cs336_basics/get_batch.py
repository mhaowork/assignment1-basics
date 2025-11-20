from multiprocessing import context
import torch
import numpy.typing as npt

def get_batch(
  dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
  indices = torch.randint(
    low=0,
    high=len(dataset) - context_length,
    size=(batch_size,),
  )

  offset = torch.arange(context_length)
  batches = dataset[indices.reshape(-1, 1) + offset]
  targets = dataset[indices.reshape(-1, 1) + offset + 1]

  return (
    torch.tensor(batches, device=device, dtype=torch.int64),
    torch.tensor(targets, device=device, dtype=torch.int64)
  )


  

  
