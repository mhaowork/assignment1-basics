import os
from torch import no_grad
import numpy as np

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import calc_cross_entropy
from cs336_basics.get_batch import get_batch
from cs336_basics.gradient_clipping import do_gradient_clipping
from cs336_basics.letstokenize import get_vocab_size
from cs336_basics.transformer_lm import TransformerLM

def train(
  train_token_file: str | os.PathLike,
  valid_token_file: str | os.PathLike,
  epoches: int,
  vocab_size: int,
  batch_size: int,
  context_length: int,
  d_model: int,
  num_layers: int,
  num_heads: int,
  d_ff: int,
  rope_theta: float,
  device: str,
  train_steps_per_epoch: int = 100,
  val_steps: int = 10,
):
  train_dataset = np.memmap(train_token_file, dtype='uint16', mode='r')
  valid_dataset = np.memmap(valid_token_file, dtype='uint16', mode='r')

  model = TransformerLM(
    vocab_size=vocab_size,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    max_seq_len=context_length,
    rope_theta=rope_theta,
    device=device,
  )
  optimizer = AdamW(params=model.parameters())


  for epoch in range(epoches):
    model.train()
    epoch_loss = 0.0
    
    # Training loop - get a new batch for each step
    for step in range(train_steps_per_epoch):
      batch_inputs, batch_targets = get_batch(dataset=train_dataset, batch_size=batch_size, context_length=context_length, device=device)
      
      optimizer.zero_grad()

      logits = model(batch_inputs) # shape [batch_size, seq_len, vocab_size]

      loss = calc_cross_entropy(
        logits=logits.view(-1, vocab_size),
        targets=batch_targets.view(-1),
      )
      loss.backward() # compute grads

      # do_gradient_clipping(model.parameters(), max_l2_norm=1.0)

      optimizer.step()
      
      epoch_loss += loss.item()
      
      print(f"  Step {step + 1}/{train_steps_per_epoch}, Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / train_steps_per_epoch
    print(f"Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    with no_grad():
      val_loss = 0.0
      for step in range(val_steps):
        batch_valid_inputs, batch_valid_targets = get_batch(dataset=valid_dataset, batch_size=batch_size, context_length=context_length, device=device)

        logits = model(batch_valid_inputs)

        loss = calc_cross_entropy(
          logits=logits.view(-1, vocab_size),
          targets=batch_valid_targets.view(-1),
        )

        val_loss += loss.item()
      
      avg_val_loss = val_loss / val_steps
      print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
  train(
    train_token_file="../data/TinyStoriesV2-GPT4-train-tokens.txt",
    valid_token_file="../data/TinyStoriesV2-GPT4-valid-tokens.txt",
    epoches=10,
    vocab_size=get_vocab_size(),
    batch_size=64,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000,
    device='mps',
    train_steps_per_epoch=100,  # Number of training batches per epoch
    val_steps=10,  # Number of validation batches
  )









  



