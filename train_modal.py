"""Modal entrypoints for training TransformerLM on a GPU worker."""

import datetime
import sys
from pathlib import Path

REPO_MOUNT_PATH = "/root/project"
if REPO_MOUNT_PATH not in sys.path:
  sys.path.append(REPO_MOUNT_PATH)

import modal

from cs336_basics.model_config import ModelConfig
from cs336_basics.train import TrainingPaths, WandbSettings, train_with_config


APP_NAME = "cs336-basics-train"
DATA_VOLUME_NAME = "cs336-basics-data"
DATA_MOUNT_PATH = Path("/data")

TRAIN_TOKEN_DEFAULT = "owt_train_tokens.txt"
VALID_TOKEN_DEFAULT = "owt_valid_tokens.txt"
DEFAULT_CHECKPOINT_PREFIX = "owt-training-checkpoint"

TORCH_VERSION = "2.4.1"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

BASE_DEPENDENCIES = [
  "einops>=0.8.1",
  "einx>=0.3.0",
  "jaxtyping>=0.3.0",
  "numpy",
  "psutil>=6.1.1",
  "regex>=2024.11.6",
  "tiktoken>=0.9.0",
  "tqdm>=4.67.1",
  "ty>=0.0.1a16",
  "wandb>=0.19.7",
]


app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

try:
  wandb_secret = modal.Secret.from_name("wandb")
except Exception:
  # Secret is optional; fallback to environment variables if not defined.
  wandb_secret = None

image = (
  modal.Image.debian_slim(python_version="3.11")
  .pip_install(f"torch=={TORCH_VERSION}", extra_index_url=TORCH_INDEX_URL)
  .pip_install(*BASE_DEPENDENCIES)
  .add_local_dir(".", remote_path=REPO_MOUNT_PATH)
)

function_kwargs = dict(
  image=image,
  gpu=modal.gpu.A100(size="80GB"),
  volumes={str(DATA_MOUNT_PATH): data_volume},
  timeout=60 * 60 * 12,
)
if wandb_secret is not None:
  function_kwargs["secrets"] = [wandb_secret]

def _build_modal_paths(
  train_filename: str,
  valid_filename: str,
  checkpoint_filename: str,
  resume_checkpoint: str | None,
) -> TrainingPaths:
  train_path = DATA_MOUNT_PATH / train_filename
  valid_path = DATA_MOUNT_PATH / valid_filename
  return TrainingPaths(
    train_token_file=str(train_path),
    valid_token_file=str(valid_path),
    out_checkpoint_file=str(DATA_MOUNT_PATH / checkpoint_filename),
    in_checkpoint_file=(
      str(DATA_MOUNT_PATH / resume_checkpoint) if resume_checkpoint is not None else None
    ),
  )


@app.function(**function_kwargs)
def run_training(
  *,
  epochs: int = 200,
  train_steps_per_epoch: int = 25,
  val_steps: int = 3,
  batch_size: int | None = None,
  train_filename: str = TRAIN_TOKEN_DEFAULT,
  valid_filename: str = VALID_TOKEN_DEFAULT,
  checkpoint_prefix: str = DEFAULT_CHECKPOINT_PREFIX,
  checkpoint_suffix: str | None = None,
  resume_checkpoint: str | None = None,
  wandb_project: str | None = None,
  wandb_run_name: str | None = None,
):
  """Launch the long-running training loop on a Modal GPU worker."""

  import sys

  if REPO_MOUNT_PATH not in sys.path:
    sys.path.append(REPO_MOUNT_PATH)

  timestamp = datetime.datetime.now().strftime("%m%d%H%M")
  checkpoint_name = checkpoint_suffix or f"{checkpoint_prefix}-{timestamp}.pt"

  wandb_settings = (
    WandbSettings(project=wandb_project, run_name=wandb_run_name or f"modal-run-{timestamp}")
    if wandb_project is not None else None
  )

  config = ModelConfig()
  paths = _build_modal_paths(
    train_filename=train_filename,
    valid_filename=valid_filename,
    checkpoint_filename=checkpoint_name,
    resume_checkpoint=resume_checkpoint,
  )

  train_with_config(
    config=config,
    paths=paths,
    epoches=epochs,
    device="cuda",
    train_steps_per_epoch=train_steps_per_epoch,
    val_steps=val_steps,
    batch_size=batch_size,
    wandb_settings=wandb_settings,
  )


@app.local_entrypoint()
def main(
  epochs: int = 200,
  train_steps_per_epoch: int = 25,
  val_steps: int = 3,
  batch_size: int | None = None,
  train_filename: str = TRAIN_TOKEN_DEFAULT,
  valid_filename: str = VALID_TOKEN_DEFAULT,
  checkpoint_suffix: str | None = None,
  resume_checkpoint: str | None = None,
  wandb_project: str | None = None,
  wandb_run_name: str | None = None,
):
  """Convenience entrypoint: `modal run train_modal.py`."""

  run_training.remote(
    epochs=epochs,
    train_steps_per_epoch=train_steps_per_epoch,
    val_steps=val_steps,
    batch_size=batch_size,
    train_filename=train_filename,
    valid_filename=valid_filename,
    checkpoint_suffix=checkpoint_suffix,
    resume_checkpoint=resume_checkpoint,
    wandb_project=wandb_project,
    wandb_run_name=wandb_run_name,
  )
