import os
import datetime
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from torch import no_grad

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import calc_cross_entropy
from cs336_basics.get_batch import get_batch
from cs336_basics.gradient_clipping import do_gradient_clipping
from cs336_basics.letstokenize import get_vocab_size
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.model_config import ModelConfig
from cs336_basics.transformer_lm import TransformerLM


@dataclass
class TrainingPaths:
  train_token_file: str | os.PathLike
  valid_token_file: str | os.PathLike
  out_checkpoint_file: str | os.PathLike
  in_checkpoint_file: str | os.PathLike | None = None


@dataclass
class WandbSettings:
  project: str
  run_name: str | None = None


def train(
  *,
  train_dataset,
  valid_dataset,
  epoches: int,
  vocab_size: int,
  batch_size: int,
  context_length: int,
  d_model: int,
  num_layers: int,
  num_heads: int,
  d_ff: int,
  rope_theta: float,
  lr_max: float,
  lr_min: float,
  warm_up_steps: int,
  annealing_steps: int,
  gradient_accumulation_steps: int = 1,
  device: str,
  out_checkpoint_file: str | os.PathLike,
  train_steps_per_epoch: int = 100,
  val_steps: int = 10,
  in_checkpoint_file: str | os.PathLike | None = None,
  wandb_settings: WandbSettings | None = None,
):

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
  optimizer = AdamW(params=model.parameters(), lr=lr_max)
  gradient_accumulation_steps = max(1, gradient_accumulation_steps)
  updates_per_epoch = (train_steps_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
  final_group_size = (
    train_steps_per_epoch % gradient_accumulation_steps
    if train_steps_per_epoch % gradient_accumulation_steps != 0
    else gradient_accumulation_steps
  )

  wandb_run = None
  if wandb_settings is not None:
    wandb_config = dict(
      train_token_file=str(getattr(train_dataset, 'filename', 'in-memory')),
      valid_token_file=str(getattr(valid_dataset, 'filename', 'in-memory')),
      batch_size=batch_size,
      context_length=context_length,
      d_model=d_model,
      num_layers=num_layers,
      num_heads=num_heads,
      d_ff=d_ff,
      rope_theta=rope_theta,
      vocab_size=vocab_size,
      train_steps_per_epoch=train_steps_per_epoch,
      val_steps=val_steps,
      lr_max=lr_max,
      lr_min=lr_min,
      warm_up_steps=warm_up_steps,
      annealing_steps=annealing_steps,
      gradient_accumulation_steps=gradient_accumulation_steps,
      device=device,
    )
    wandb_run = wandb.init(
      project=wandb_settings.project,
      name=wandb_settings.run_name,
      config=wandb_config,
    )
    wandb_run.define_metric("global_step")
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("train/*", step_metric="global_step")
    wandb_run.define_metric("val/*", step_metric="global_step")

  # Track which epoch/iteration we are resuming from so checkpoints can
  # restart where training left off.
  epoch = 0
  if in_checkpoint_file is not None:
    print("Loading model & optimizer from ", in_checkpoint_file)
    epoch = load_checkpoint(
      src=in_checkpoint_file,
      model=model,
      optimizer=optimizer,
    )

  should_stop = False
  try:
    while epoch < epoches:
      print(f"=========================Starting Epoch {epoch}==================")
      if should_stop:
        break
      model.train()
      epoch_loss = 0.0
      accum_loss = 0.0
      accum_count = 0
      update_counter = 0
      global_update_step = epoch * updates_per_epoch
      optimizer.zero_grad()
      
      # Training loop - get a new batch for each step
      for step in range(train_steps_per_epoch):
        accumulation_edge = ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == train_steps_per_epoch)

        batch_inputs, batch_targets = get_batch(dataset=train_dataset, batch_size=batch_size, context_length=context_length, device=device)

        logits = model(batch_inputs) # shape [batch_size, seq_len, vocab_size]

        loss = calc_cross_entropy(
          logits=logits.view(-1, vocab_size),
          targets=batch_targets.view(-1),
        )
        loss_item = loss.item()
        current_group_index = step // gradient_accumulation_steps
        current_group_size = (
          final_group_size if current_group_index == updates_per_epoch - 1 else gradient_accumulation_steps
        )
        loss = loss / current_group_size
        loss.backward() # compute grads
        accum_loss += loss_item
        accum_count += 1
        epoch_loss += loss_item

        if device == 'mps' and torch.backends.mps.is_available():
          current_mem = torch.mps.current_allocated_memory() / (1024 ** 2)
          print(f"    MPS allocated memory: {current_mem:.2f} MB")

        if accumulation_edge:
          do_gradient_clipping(model.parameters(), max_l2_norm=1.0)
          update_counter += 1
          global_update_step += 1
          lr = get_lr_cosine_schedule(
              t=global_update_step,
              lr_max=lr_max,
              lr_min=lr_min,
              warm_up_steps=warm_up_steps,
              annealing_steps=annealing_steps
          )
          for param_group in optimizer.param_groups:
              param_group['lr'] = lr
          optimizer.step()
          optimizer.zero_grad()
          avg_accum_loss = accum_loss / accum_count
          accum_loss = 0.0
          accum_count = 0
        
          print(f"  Step {step + 1}/{train_steps_per_epoch} (update {global_update_step}), Loss: {avg_accum_loss:.4f}, LR: {lr:.6f}")

          if wandb_run is not None:
            wandb_run.log(
              {
                "train/loss": avg_accum_loss,
                "train/lr": lr,
                "train/step": update_counter,
                "epoch": epoch,
                "global_step": global_update_step,
              },
              step=global_update_step,
            )

      avg_train_loss = epoch_loss / train_steps_per_epoch
      print(f"Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}")
      if wandb_run is not None:
        wandb_run.log(
          {
            "train/avg_epoch_loss": avg_train_loss,
            "epoch": epoch,
            "global_step": global_update_step,
          },
          step=global_update_step,
        )
      
      # Validation loop
      model.eval()
      with no_grad():
        if should_stop:
          break
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
        if wandb_run is not None:
          wandb_run.log(
            {
              "val/loss": avg_val_loss,
              "epoch": epoch,
              "global_step": global_update_step,
            },
            step=global_update_step,
          )
      epoch += 1
  except KeyboardInterrupt:
    should_stop = True
    print('\nExiting')

  print("Saving checkpoint to ", out_checkpoint_file) 
  save_checkpoint(
    model=model,
    optimizer=optimizer,
    iteration=epoch,
    out=out_checkpoint_file,
  )
  if wandb_run is not None:
    wandb_run.finish()


def train_with_config(
  config: ModelConfig,
  paths: TrainingPaths,
  *,
  epoches: int,
  device: str,
  train_steps_per_epoch: int = 100,
  val_steps: int = 10,
  batch_size: int | None = None,
  gradient_accumulation_steps: int | None = None,
  wandb_settings: WandbSettings | None = None,
  train_dataset=None,
  valid_dataset=None,
):
  """Run training using a ModelConfig and filesystem paths."""

  train_data = train_dataset
  if train_data is None:
    train_data = np.memmap(paths.train_token_file, dtype='uint16', mode='r')

  valid_data = valid_dataset
  if valid_data is None:
    valid_data = np.memmap(paths.valid_token_file, dtype='uint16', mode='r')

  train(
    train_dataset=train_data,
    valid_dataset=valid_data,
    epoches=epoches,
    vocab_size=config.vocab_size,
    batch_size=batch_size or config.batch_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    rope_theta=config.rope_theta,
    lr_max=config.lr_max,
    lr_min=config.lr_min,
    warm_up_steps=config.warm_up_steps,
    annealing_steps=config.annealing_steps,
    gradient_accumulation_steps=gradient_accumulation_steps or config.gradient_accumulation_steps,
    device=device,
    out_checkpoint_file=paths.out_checkpoint_file,
    train_steps_per_epoch=train_steps_per_epoch,
    val_steps=val_steps,
    in_checkpoint_file=paths.in_checkpoint_file,
    wandb_settings=wandb_settings,
  )


if __name__ == "__main__":
  timestamp = datetime.datetime.now().strftime("%m%d%H%M")
  config = ModelConfig()
  train_with_config(
    config=config,
    paths=TrainingPaths(
      train_token_file="../data/owt_train_tokens.txt",
      valid_token_file="../data/owt_valid_tokens.txt",
      out_checkpoint_file=f"../data/owt-checkpoint-{timestamp}.txt",
      # in_checkpoint_file="../data/TinyStoriesV2-GPT4-checkpoint-11192009.txt",
    ),
    epoches=200,
    device='mps',
    train_steps_per_epoch=25,  # Number of training batches per epoch
    val_steps=3,  # Number of validation batches
    wandb_settings=WandbSettings(
      project="cs336-assignment1",
      run_name=f"owt-training-{timestamp}",
    ),
  )









  
