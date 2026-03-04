"""
dataset.py
==========
Converts a token sequence into training triplets.
"""

import numpy as np
from src.vocabulary import Vocabulary


def _save_training_pairs(pairs, filename: str, variant: str):
    """
    Save training pairs to a file.
    """
    with open(filename, "w") as f:
        if variant == "sgns":
            for center_id, context_id, neg_ids in pairs:
                neg_str = " ".join(map(str, neg_ids))
                f.write(f"{center_id} {context_id} {neg_str}\n")
        else:
            for target_id, context_ids, neg_ids in pairs:
                context_str = " ".join(map(str, context_ids))
                neg_str = " ".join(map(str, neg_ids))
                line = f"{target_id} {context_str} | {neg_str}\n"
                f.write(line)


def _load_training_pairs(filename: str, variant: str):
    """
    Load training pairs from a file. Expects the same format as saved by save_training_pairs.
    """
    pairs = []
    with open(filename, "r") as f:
        if variant == "sgns":
             for line in f:
                parts = line.strip().split()
                center_id = int(parts[0])
                context_id = int(parts[1])
                neg_ids = list(map(int, parts[2:]))
                pairs.append((center_id, context_id, neg_ids))
        else:
            for line in f:
                target_part, neg_part = line.strip().split("|")
                target_id, *context_ids = map(int, target_part.strip().split())
                neg_ids = list(map(int, neg_part.strip().split()))
                pairs.append((target_id, context_ids, neg_ids))
    return pairs

def _sample_negative(
    vocab_size: int,
    noise_dist: np.ndarray,
    num_negatives: int,
    exclude: set[int]
) -> list[int]:
    """Draw negative samples from the noise distribution, ensuring they are not in the exclude set."""
    neg_ids = []
    while len(neg_ids) < num_negatives:
        neg_id = np.random.choice(vocab_size, p=noise_dist)
        if neg_id not in exclude:
            neg_ids.append(neg_id)
    return neg_ids


def _build_sgns_pairs(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int,
    num_negatives: int
) -> list[tuple[int, int, list[int]]]:
    """Build training pairs for the SGNS variant."""
    pairs = []
    vocab_size = len(vocab.idx2word)
    corpus_length = len(token_ids)

    for idx in range(window_size, corpus_length - window_size):
        center_id = token_ids[idx]
        start = idx - window_size
        end = idx + window_size + 1

        for j in range(start, end):
            if j == idx:
                continue  # skip the center word itself
            context_id = token_ids[j]
            exclude = {center_id, context_id}
            neg_ids = _sample_negative(vocab_size, vocab.noise_dist, num_negatives, exclude)
            pairs.append((center_id, context_id, neg_ids))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{corpus_length - 2*window_size} tokens", end="\r")

    return pairs


def _build_cbowns_pairs(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int,
    num_negatives: int
) -> list[tuple[int, list[int], list[int]]]:
    """Build training pairs for the CBOW-NS variant."""
    pairs = []
    vocab_size = len(vocab.idx2word)
    corpus_length = len(token_ids)

    for idx in range(window_size, corpus_length - window_size):
        context_ids = token_ids[idx - window_size:idx] + token_ids[idx + 1:idx + window_size + 1]
        center_id = token_ids[idx]
        exclude = set(context_ids) | {center_id}
        neg_ids = _sample_negative(vocab_size, vocab.noise_dist, num_negatives, exclude)
        pairs.append((center_id, context_ids, neg_ids))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{corpus_length - 2*window_size} tokens", end="\r")

    return pairs


def build_training_pairs(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int = 5,
    num_negatives: int = 5,
    variant: str = "sgns"
) -> list[tuple[int, int, list[int]]]:
    """
    Returns list of (center_id, context_id, [neg_id, ...]) tuples.
    """

    pairs_filename = f"data/training_pairs/{variant}_vocab_size_{len(vocab.idx2word)}_window_size_{window_size}_num_negatives_{num_negatives}.txt"
    try:
        print(f"Loading training pairs from {pairs_filename}...")
        return _load_training_pairs(pairs_filename, variant)
    except FileNotFoundError:
        print(f"No pre-saved training pairs found. Building from scratch...")

    if variant  == "sgns":
        pairs = _build_sgns_pairs(token_ids, vocab, window_size, num_negatives)
    else:
        pairs = _build_cbowns_pairs(token_ids, vocab, window_size, num_negatives)

    print(f"\nBuilt {len(pairs)} training pairs. Saving to {pairs_filename}...")

    _save_training_pairs(pairs, pairs_filename, variant)

    return pairs
