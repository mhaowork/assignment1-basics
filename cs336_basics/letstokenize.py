import os
from typing import Sequence
import tiktoken
import numpy as np

def encode_text(text: str, model='gpt-2'):
  enc = tiktoken.encoding_for_model(model)

  return enc.encode_ordinary(text=text)

def decode_tokens(tokens: Sequence[int], model='gpt-2'):
  enc = tiktoken.encoding_for_model(model)

  return enc.decode(tokens)

def get_special_tokens(model='gpt-2'):
  return tiktoken.encoding_for_model(model).special_tokens_set

def encode(src: str | os.PathLike, dst: str | os.PathLike, model='gpt-2'):
  buffer = []
  token_count = 0
  with open(src, 'r', encoding='utf-8') as f_in,\
       open(dst, 'wb') as f_out:
    while True:
      chunk = f_in.read(1024 * 1024) # 1 MB

      if not chunk:
        break
      tokens = encode_text(text=chunk, model=model) #TODO: special tokens?
      
      buffer.extend(tokens)
      

      if len(buffer) > 1_000_000:
        assert model == 'gpt-2' # assume for now (2-byte tokens)
        arr = np.array(buffer, dtype=np.uint16)
        f_out.write(arr.tobytes())
        token_count += len(buffer)
        buffer = []
      print(f"token_count", token_count)
      
    if buffer:
      arr = np.array(buffer, dtype=np.uint16)
      f_out.write(arr.tobytes())
      token_count += len(buffer)
  print(f"Saved {token_count} tokens to {dst}")

def get_vocab_size(model='gpt-2'):
  enc = tiktoken.encoding_for_model(model)
  return enc.n_vocab

if __name__ == "__main__":
  # encode('../data/TinyStoriesV2-GPT4-train.txt', '../data/TinyStoriesV2-GPT4-train-tokens.txt')
  # encode('../data/TinyStoriesV2-GPT4-valid.txt', '../data/TinyStoriesV2-GPT4-valid-tokens.txt')
  # encode('../data/owt_train.txt', '../data/owt_train_tokens.txt')
  encode('../data/owt_valid.txt', '../data/owt_valid_tokens.txt')




  