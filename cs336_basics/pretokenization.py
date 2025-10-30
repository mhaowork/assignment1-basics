from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
import os
import regex as re
from typing import BinaryIO


def pre_tokenize_chunk(chunk: str, special_tokens: list[bytes]) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokens = {}
    # Decode special tokens to strings to match chunk type
    decoded_special_tokens = [token.decode("utf-8") for token in special_tokens]
    escaped_special_tokens = [re.escape(token) for token in decoded_special_tokens]
    subchunks = re.split("|".join(escaped_special_tokens), chunk)
    for subchunk in subchunks:
        for match in re.finditer(PAT, subchunk):
            pretokens[match.group()] = pretokens.get(match.group(), 0) + 1
    return pretokens

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_file(input_path: str, num_processes: int, special_token: bytes) -> dict[str, int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)

        pretoken_count_batches= []
        chunks = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

        print('chunking done')
        # Run pre-tokenization on each chunk in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            pretoken_count_batches = list(executor.map(pre_tokenize_chunk, chunks, repeat([special_token])))
        
        pretoken_count: dict[str, int] = {}
        for batch in pretoken_count_batches:
            for pretoken, count in batch.items():
                    # pretoken_in_ints = tuple[int]([int(b) for b in pretoken.encode("utf-8")])
                    pretoken_count[pretoken] = pretoken_count.get(pretoken, 0) + count
        return pretoken_count

if __name__ == "__main__":
    pretoken_count = pretokenize_file('data/TinyStoriesV2-GPT4-valid.txt', 8, b"<|endoftext|>")
    print(pretoken_count)