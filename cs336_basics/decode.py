import os
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.model_config import ModelConfig
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.softmax import softmax
import letstokenize


def generate_text(
    checkpoint_file: str | os.PathLike,
    prompt_text: str,
    max_generated_len: int,
    temperature: float,
    top_p: int,
    device: torch.device,
):
    config = ModelConfig()
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.context_length,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
    )

    load_checkpoint(
    src=checkpoint_file,
    model=model,
    optimizer=None,
  )

    curr_tokens = letstokenize.encode_text(prompt_text)
    with torch.no_grad():
        for _ in range(max_generated_len):
            print('curr_tokens', curr_tokens)
            logits = model(torch.tensor([curr_tokens], device=device))

            logits /= temperature

            # TODO: impl top p
            last_logits = logits[:, -1, :]
            probs = softmax(last_logits, dim_idx=-1)[0] # [seq_len ]

            sorted_probs, sorted_probs_indices = torch.sort(probs, descending=True)

            top_p_probs = torch.zeros_like(probs)
            top_p_probs[sorted_probs_indices[: top_p]] = sorted_probs[: top_p]

            print('prob', probs)
            print('top_p_prob', top_p_probs)
            print('sorted_probs', sorted_probs)

            next_token = torch.multinomial(top_p_probs, num_samples=1)
            next_token = next_token[0].item()

            print('Generated: ', letstokenize.decode_tokens([next_token]))
            curr_tokens.append(next_token)
            
            if next_token in letstokenize.get_special_tokens():
              break

    print("Finished: ", letstokenize.decode_tokens(curr_tokens))


if __name__ == "__main__":
  prompt = "This is the last word. Bye"
  generate_text(
    checkpoint_file='../data/TinyStoriesV2-GPT4-checkpoint-11200908.txt',
    prompt_text=prompt,
    max_generated_len=105,
    temperature=0.5,
    top_p=5,
    device='mps',
  )
