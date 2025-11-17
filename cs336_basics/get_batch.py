from multiprocessing import context
import torch
import numpy.typing as npt

def get_batch(
  dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
  t = torch.Tensor(dataset, device=device)
  
  #TODO: what if dataset is too big to load (see assignment doc)

  indices = torch.randint(
    low=0,
    high=len(dataset) - context_length,
    size=(batch_size,),
    device=device,
  )

  offset = torch.arange(context_length, device=indices.device)
  batches = t[indices.reshape(-1, 1) + offset]
  targets = t[indices.reshape(-1, 1) + offset + 1]

  return batches, targets


  

  
