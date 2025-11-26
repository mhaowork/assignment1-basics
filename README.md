# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training on Modal GPUs

You can launch the provided training loop on a hosted GPU using
[Modal](https://modal.com/). The `train_modal.py` script packages this repo,
mounts a persistent volume that stores tokenizer data/checkpoints, and runs the
`train_with_config` helper inside a GPU function.

1. Install the Modal CLI (`pip install modal`) and log in with
   `modal token new`.
2. Create a volume that will hold the tokenized training data (defaults to the
   OpenWebText tokens) and any checkpoints. The script defaults to
   `cs336-basics-data` mounted at `/data`. Upload files with:

   ```sh
   modal volume create cs336-basics-data  # only once
   modal volume put cs336-basics-data data/owt_train_tokens.txt:/owt_train_tokens.txt
   modal volume put cs336-basics-data data/owt_valid_tokens.txt:/owt_valid_tokens.txt
   ```

3. (Optional) Store your WandB API key in a Modal secret so the remote job can
   log metrics:

   ```sh
   modal secret create wandb WANDB_API_KEY=...your key...
   ```

4. Launch training. The defaults mirror the local `train.py` main block, but
   you can pass flags to override epochs, batch size, checkpoint names, etc.:

   ```sh
   modal run train_modal.py --epochs 200 --train_steps_per_epoch 25 --val_steps 3 --wandb_project cs336-assignment1
   ```

   Checkpoints are saved back into the same volume (e.g.
   `/data/owt-training-checkpoint-<timestamp>.pt`). Retrieval uses the
   CLI: `modal volume get cs336-basics-data owt-training-checkpoint-<ts>.pt .`.

The Modal function requests an A100 (80GB) GPU, installs the repo dependencies, and
invokes `train_with_config` with `device="cuda"`. Adjust the GPU type or image
dependencies in `train_modal.py` if your account has access to different
hardware.
