from collections import defaultdict
from functools import cmp_to_key
import os

from sympy import O
from sympy.sets.contains import Set
from torch._dynamo.utils import count_calls
from tests.common import gpt2_bytes_to_unicode

from cs336_basics.pretokenization import pretokenize_file
from sortedcontainers import SortedDict, SortedSet



def get_initial_vocab() -> dict[int, bytes]:
  # TODO: DO NOT hardcode the endoftext special token
  return {0: b'<|endoftext|>', 1: b'!', 2: b'"', 3: b'#', 4: b'$', 5: b'%', 6: b'&', 7: b"'", 8: b'(', 9: b')', 10: b'*', 11: b'+', 12: b',', 13: b'-', 14: b'.', 15: b'/', 16: b'0', 17: b'1', 18: b'2', 19: b'3', 20: b'4', 21: b'5', 22: b'6', 23: b'7', 24: b'8', 25: b'9', 26: b':', 27: b';', 28: b'<', 29: b'=', 30: b'>', 31: b'?', 32: b'@', 33: b'A', 34: b'B', 35: b'C', 36: b'D', 37: b'E', 38: b'F', 39: b'G', 40: b'H', 41: b'I', 42: b'J', 43: b'K', 44: b'L', 45: b'M', 46: b'N', 47: b'O', 48: b'P', 49: b'Q', 50: b'R', 51: b'S', 52: b'T', 53: b'U', 54: b'V', 55: b'W', 56: b'X', 57: b'Y', 58: b'Z', 59: b'[', 60: b'\\', 61: b']', 62: b'^', 63: b'_', 64: b'`', 65: b'a', 66: b'b', 67: b'c', 68: b'd', 69: b'e', 70: b'f', 71: b'g', 72: b'h', 73: b'i', 74: b'j', 75: b'k', 76: b'l', 77: b'm', 78: b'n', 79: b'o', 80: b'p', 81: b'q', 82: b'r', 83: b's', 84: b't', 85: b'u', 86: b'v', 87: b'w', 88: b'x', 89: b'y', 90: b'z', 91: b'{', 92: b'|', 93: b'}', 94: b'~', 95: b'\xa1', 96: b'\xa2', 97: b'\xa3', 98: b'\xa4', 99: b'\xa5', 100: b'\xa6', 101: b'\xa7', 102: b'\xa8', 103: b'\xa9', 104: b'\xaa', 105: b'\xab', 106: b'\xac', 107: b'\xae', 108: b'\xaf', 109: b'\xb0', 110: b'\xb1', 111: b'\xb2', 112: b'\xb3', 113: b'\xb4', 114: b'\xb5', 115: b'\xb6', 116: b'\xb7', 117: b'\xb8', 118: b'\xb9', 119: b'\xba', 120: b'\xbb', 121: b'\xbc', 122: b'\xbd', 123: b'\xbe', 124: b'\xbf', 125: b'\xc0', 126: b'\xc1', 127: b'\xc2', 128: b'\xc3', 129: b'\xc4', 130: b'\xc5', 131: b'\xc6', 132: b'\xc7', 133: b'\xc8', 134: b'\xc9', 135: b'\xca', 136: b'\xcb', 137: b'\xcc', 138: b'\xcd', 139: b'\xce', 140: b'\xcf', 141: b'\xd0', 142: b'\xd1', 143: b'\xd2', 144: b'\xd3', 145: b'\xd4', 146: b'\xd5', 147: b'\xd6', 148: b'\xd7', 149: b'\xd8', 150: b'\xd9', 151: b'\xda', 152: b'\xdb', 153: b'\xdc', 154: b'\xdd', 155: b'\xde', 156: b'\xdf', 157: b'\xe0', 158: b'\xe1', 159: b'\xe2', 160: b'\xe3', 161: b'\xe4', 162: b'\xe5', 163: b'\xe6', 164: b'\xe7', 165: b'\xe8', 166: b'\xe9', 167: b'\xea', 168: b'\xeb', 169: b'\xec', 170: b'\xed', 171: b'\xee', 172: b'\xef', 173: b'\xf0', 174: b'\xf1', 175: b'\xf2', 176: b'\xf3', 177: b'\xf4', 178: b'\xf5', 179: b'\xf6', 180: b'\xf7', 181: b'\xf8', 182: b'\xf9', 183: b'\xfa', 184: b'\xfb', 185: b'\xfc', 186: b'\xfd', 187: b'\xfe', 188: b'\xff', 189: b'\x00', 190: b'\x01', 191: b'\x02', 192: b'\x03', 193: b'\x04', 194: b'\x05', 195: b'\x06', 196: b'\x07', 197: b'\x08', 198: b'\t', 199: b'\n', 200: b'\x0b', 201: b'\x0c', 202: b'\r', 203: b'\x0e', 204: b'\x0f', 205: b'\x10', 206: b'\x11', 207: b'\x12', 208: b'\x13', 209: b'\x14', 210: b'\x15', 211: b'\x16', 212: b'\x17', 213: b'\x18', 214: b'\x19', 215: b'\x1a', 216: b'\x1b', 217: b'\x1c', 218: b'\x1d', 219: b'\x1e', 220: b'\x1f', 221: b' ', 222: b'\x7f', 223: b'\x80', 224: b'\x81', 225: b'\x82', 226: b'\x83', 227: b'\x84', 228: b'\x85', 229: b'\x86', 230: b'\x87', 231: b'\x88', 232: b'\x89', 233: b'\x8a', 234: b'\x8b', 235: b'\x8c', 236: b'\x8d', 237: b'\x8e', 238: b'\x8f', 239: b'\x90', 240: b'\x91', 241: b'\x92', 242: b'\x93', 243: b'\x94', 244: b'\x95', 245: b'\x96', 246: b'\x97', 247: b'\x98', 248: b'\x99', 249: b'\x9a', 250: b'\x9b', 251: b'\x9c', 252: b'\x9d', 253: b'\x9e', 254: b'\x9f', 255: b'\xa0', 256: b'\xad'}

def print_pair_count(vocab: dict[int, bytes], pair_count: dict[tuple[int, int], int]) -> None:
  for pair, count in pair_count.items():
    print(f'{vocab[pair[0]]} {vocab[pair[1]]} {count}')

def print_token_count(vocab: dict[int, bytes], token_count: dict[tuple[int], int]) -> None:
  print('token count:')
  for token, count in token_count.items():
    print(f'{[vocab[i] for i in token]} {count}')

  


def train_bpe_naive(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  """Given the path to an input corpus, run train a BPE tokenizer and
  output its vocabulary and merges.

  Args:
      input_path (str | os.PathLike): Path to BPE tokenizer training data.
      vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
      special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
          These strings will never be split into multiple tokens, and will always be
          kept as a single token. If these special tokens occur in the `input_path`,
          they are treated as any other string.

  Returns:
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
          vocab:
              The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
              to bytes (token bytes)
          merges:
              BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
              representing that <token1> was merged with <token2>.
              Merges are ordered by order of creation.
  """


  def compare_pairs(pair1: tuple[tuple[int, int], int], pair2: tuple[tuple[int, int], int]) -> int:
    if pair1[1] != pair2[1]:
      return 1 if pair1[1] < pair2[1] else -1
    str1 = vocab[pair1[0][0]]
    str2 = vocab[pair2[0][0]]
    if str1 != str2:
      return 1 if str1 < str2 else -1
    return 1 if vocab[pair1[0][1]] < vocab[pair2[0][1]] else -1

  vocab: dict[int, bytes] = get_initial_vocab()
  initial_vocab_byte_to_int = {v: k for k, v in vocab.items()}
  merges: list[tuple[bytes, bytes]] = []
  pretoken_count: dict[str, int] = pretokenize_file(input_path, 8, special_tokens[0].encode("utf-8"))

  # convert bytes to tuple of ints
  token_count: dict[tuple[int], int] = {
    (tuple[int](initial_vocab_byte_to_int[bytes([b])] for b in pretoken.encode("utf-8") )): count 
    for pretoken, count in pretoken_count.items()
  }

  for _ in range(vocab_size - 256 - len(special_tokens)):
    pair_count: dict[tuple[int, int], int] = {}
    for token, count in token_count.items():
      for i in range(len(token) - 1):
        pair = (token[i], token[i + 1])
        pair_count[pair] = pair_count.get(pair, 0) + count
      
    print('token count length', len(token_count))
  
    sorted_pairs = sorted(pair_count.items(), key=cmp_to_key(compare_pairs))

    # print('first in sorted_pairs', [(vocab[pair[0][0]] + vocab[pair[0][1]], pair[1]) for pair in sorted_pairs[:5]])

    # print_token_count(vocab, token_count)
    # print_pair_count(vocab, pair_count)

    pair_to_merge: tuple[int, int] = sorted_pairs[0][0]

    # print('pair to merge', vocab[pair_to_merge[0]], vocab[pair_to_merge[1]], sorted_pairs[0][1])
    new_vocab_element = len(vocab)

    # update token count
    new_token_count: dict[tuple[int], int] = {}
    for token, count in token_count.items():
      new_token = list[int]()
      # a b c d where we want to merge b and c as new token e
      # => a e d
      idx = 0
      while idx < len(token):
        if idx == len(token) - 1:
          new_token.append(token[idx])
          break
        pair = (token[idx], token[idx + 1])
        if pair == pair_to_merge:
          new_token.append(new_vocab_element)
          idx += 2
        else:
          new_token.append(token[idx])
          idx += 1
      new_token_count[tuple[int](new_token)] = count
    token_count = new_token_count
    vocab[new_vocab_element] = vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]]
    merges.append((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]))

  return vocab, merges



def calc_pair_count(token: tuple[int]) -> dict[tuple[int, int], int]:
  pair_count: dict[tuple[int, int], int] = {}
  for idx in range(len(token) - 1):
    pair_count[(token[idx], token[idx + 1])] = pair_count.get((token[idx], token[idx + 1]), 0) + 1
  return pair_count

def transform_token(token: tuple[int], pair_to_merge: tuple[int, int], new_elem: int) -> tuple[int]:
  new_token = []
  idx = 0
  while idx < len(token):
    if idx == len(token) - 1:
      new_token.append(token[idx])
      break
    if token[idx] == pair_to_merge[0] and token[idx + 1] == pair_to_merge[1]:
      new_token.append(new_elem)
      idx += 2
    else:
      new_token.append(token[idx])
      idx += 1
  return tuple(new_token)


def compute_delta(token: tuple[int], pair_to_merge: tuple[int, int], new_elem: int):
  old_pair_count = calc_pair_count(token)

  new_token = transform_token(token, pair_to_merge, new_elem)

  new_pair_count = calc_pair_count(new_token)

  deltas: dict[tuple[int, int], tuple[int, bool]] = {}
  should_add_or_remove = False
  for pair in set(old_pair_count.keys()) | set(new_pair_count.keys()):
    old_count, new_count = old_pair_count.get(pair, 0), new_pair_count.get(pair, 0)

    if old_count == new_count:
      continue

    should_add_or_remove = new_count == 0 or old_count == 0
    deltas[pair] = (new_count - old_count, should_add_or_remove)

  return deltas



def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  """Given the path to an input corpus, run train a BPE tokenizer and
  output its vocabulary and merges.

  Args:
      input_path (str | os.PathLike): Path to BPE tokenizer training data.
      vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
      special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
          These strings will never be split into multiple tokens, and will always be
          kept as a single token. If these special tokens occur in the `input_path`,
          they are treated as any other string.

  Returns:
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
          vocab:
              The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
              to bytes (token bytes)
          merges:
              BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
              representing that <token1> was merged with <token2>.
              Merges are ordered by order of creation.
  """


  vocab: dict[int, bytes] = get_initial_vocab()
  initial_vocab_byte_to_int = {v: k for k, v in vocab.items()}
  merges: list[tuple[bytes, bytes]] = []

  def compare_pairs(pair1: tuple[int, int], pair2: tuple[int, int]) -> int:
    str1 = vocab[pair1[0]]
    str2 = vocab[pair2[0]]
    if str1 != str2:
      return 1 if str1 < str2 else -1
    str1 = vocab[pair1[1]]
    str2 = vocab[pair2[1]]
    if str1 != str2:
      return 1 if str1 < str2 else -1
    return 0

  pretoken_count: dict[str, int] = pretokenize_file(input_path, 1, special_tokens[0].encode("utf-8"))

  # convert bytes to tuple of ints
  token_count: dict[tuple[int], int] = {
    (tuple[int](initial_vocab_byte_to_int[bytes([b])] for b in pretoken.encode("utf-8") )): count 
    for pretoken, count in pretoken_count.items()
  }
  pair_count: dict[tuple[int, int], int] = {}
  pair_to_token_ids: dict[tuple[int, int], set[int]] = defaultdict(set)

  id_to_token: dict[int, tuple[int]] = {}
  token_id_to_count: dict[int, int] = {}
  
  token_id = 0
  for token, count in token_count.items():
    # print('token', token, 'count', count)
    id_to_token[token_id] = token
    token_id_to_count[token_id] = count
    for i in range(len(token) - 1):
      pair = (token[i], token[i + 1])
      pair_count[pair] = pair_count.get(pair, 0) + count
      pair_to_token_ids[pair].add(token_id)
    token_id += 1

  # print('counts', pair_count)
  # print(pair_to_tokens)

  # build count -> pair ss(sorted set)
  #count_to_pair_ss: SortedDict[int, SortedSet[tuple[int, int]]] = defaultdict(lambda: SortedSet(key=cmp_to_key(compare_pairs)))
  count_to_pair_ss: SortedDict[int, SortedSet[tuple[int, int]]] = SortedDict()
  max_count = 0
  for pair, count in pair_count.items():
    if not count_to_pair_ss.get(count):
      count_to_pair_ss[count] = SortedSet([], key=cmp_to_key(compare_pairs))
    count_to_pair_ss[count].add(pair)
    max_count = max(count, max_count)

  # print("max_count", max_count, count_to_pair_ss)

  new_elem = len(vocab) - 1
  while len(vocab) < vocab_size:
    new_elem += 1

    max_count = count_to_pair_ss.peekitem(-1)[0]
    while len(count_to_pair_ss[max_count]) == 0:
      count_to_pair_ss.popitem()
      max_count = count_to_pair_ss.peekitem(-1)[0]
    pair_to_merge: tuple[int, int]= count_to_pair_ss[max_count][0]

    # print('pair_to_merge', pair_to_merge, vocab[pair_to_merge[0]], vocab[pair_to_merge[1]])

    # new_tokens = []

    left_word, right_word = vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]
    vocab[new_elem] = left_word + right_word
    merges.append((left_word, right_word))

    affected_token_ids = pair_to_token_ids[pair_to_merge].copy()
    def count_pairs(token: tuple[int], pair: tuple[int, int]) -> int:
      count = 0
      for i in range(len(token) - 1):
        if token[i] == pair[0] and token[i + 1] == pair[1]:
          count += 1
      return count
    # print('affected_tokens', len(affected_token_ids), affected_token_ids, [token_id_to_count[token_id] * count_pairs(id_to_token[token_id], pair_to_merge) for token_id in affected_token_ids])
    for token_id in affected_token_ids:
      if token_id_to_count[token_id] == 0:
        continue
      deltas = compute_delta(id_to_token[token_id], pair_to_merge, new_elem)

      new_token = transform_token(id_to_token[token_id], pair_to_merge, new_elem)
      id_to_token[token_id] = new_token
      # print('deltas', deltas)

      for affected_pair, (count_diff, should_add_or_remove) in deltas.items():
        pair_count[affected_pair] = pair_count.get(affected_pair, 0)
        if affected_pair in count_to_pair_ss.get(pair_count[affected_pair], set()):
          count_to_pair_ss[pair_count[affected_pair]].remove(affected_pair)
        if should_add_or_remove:
          if count_diff > 0:
            pair_to_token_ids[affected_pair].add(token_id)
          else:
            pair_to_token_ids[affected_pair].remove(token_id)
        pair_count[affected_pair] += count_diff * token_id_to_count[token_id]
        if not count_to_pair_ss.get(pair_count[affected_pair]):
          count_to_pair_ss[pair_count[affected_pair]] = SortedSet([], key=cmp_to_key(compare_pairs))
        count_to_pair_ss[pair_count[affected_pair]].add(affected_pair)
      
    # pair_count.remove(pair_to_merge)
    # count_to_pair_ss[max_count].remove(pair_to_merge)
    # if pair_to_merge == (82, 69):
    #   break
  return vocab, merges



    



  


if __name__ == "__main__":
  # vocab, merges = train_bpe('data/hmtest.txt', 265, ["<|endoftext|>"])
  vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 500, ["<|endoftext|>"])
  print(vocab)
  print(merges)
