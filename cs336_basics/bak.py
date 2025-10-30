  vocab: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
  pretoken_count: dict[str, int] = pretokenize_file(input_path, 8, special_tokens[0].encode("utf-8"))

  # build byte pair -> count dict

  pair_count: dict[tuple[int, int], int] = {}
  pair_to_tokens: dict[tuple[int, int], Set[str]] = {}
  for pretoken, count in pretoken_count.items():
    pretoken_bytes: bytes = pretoken.encode("utf-8")
    for i in range(len(pretoken_bytes) - 1):
      pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
      pair_count[pair] = pair_count.get(pair, 0) + count
      pair_to_tokens[pair].add(pretoken)


  merges: list[tuple[bytes, bytes]] = []
  for _ in range(vocab_size - 256):
    # sort pairs by count
    sorted_pairs = sorted(pair_count.items(), key=lambda x: (x[1], x[0]), reverse=True)

    pair_to_merge: tuple[int, int] = sorted_pairs[0][0]
    tokens_containing_merged_pair: Set[str] = pair_to_tokens[pair_to_merge]

    for idx in range(len(vocab)):
      pair_to_test: tuple[int, int] = (idx, pair_to_merge[0])
      tokens_to_test = pair_to_tokens.get(pair_to_test)
      if tokens_to_test:
        overlapping_tokens: Set[tuple[int]] = tokens_to_test & tokens_containing_merged_pair
        new_pair = (idx, len(vocab))
        for token in overlapping_tokens:
          pair_to_tokens[pair_to_test].remove(token)
          pair_count[pair_to_test] -= pretoken_count[token]
          pair_count[new_pair] = pair_count.get(new_pair, 0) + pretoken_count[token]
          pair_to_tokens[new_pair].add(token)
    merges.append((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]))
    vocab[len(vocab)] = vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]]
    print(merges[-1])
  return vocab, merges