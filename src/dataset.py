"""
dataset.py
==========
Converts a token sequence into (center, context, negatives) training triplets.
"""

import numpy as np
from src.vocabulary import Vocabulary


def save_training_pairs(pairs: list[tuple[int, int, list[int]]], filename: str):
    """
    Save training pairs to a file. Each line will have the format:
    center_id context_id neg_id1 neg_id2 ...
    """
    with open(filename, "w") as f:
        for center_id, context_id, neg_ids in pairs:
            line = f"{center_id} {context_id} " + " ".join(map(str, neg_ids)) + "\n"
            f.write(line)


def load_training_pairs(filename: str) -> list[tuple[int, int, list[int]]]:
    """
    Load training pairs from a file. Expects the same format as saved by save_training_pairs.
    """
    pairs = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            center_id = int(parts[0])
            context_id = int(parts[1])
            neg_ids = list(map(int, parts[2:]))
            pairs.append((center_id, context_id, neg_ids))
    return pairs


def build_training_pairs(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int = 5,
    num_negatives: int = 5,
) -> list[tuple[int, int, list[int]]]:
    """
    Returns list of (center_id, context_id, [neg_id, ...]) tuples.
    """

    pairs_filename = f"data/training_pairs/vocab_size_{len(vocab.idx2word)}_window_size_{window_size}_num_negatives_{num_negatives}.txt"
    try:
        print(f"Loading training pairs from {pairs_filename}...")
        return load_training_pairs(pairs_filename)
    except FileNotFoundError:
        print(f"No pre-saved training pairs found. Building from scratch...")

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
            neg_ids = []
            while len(neg_ids) < num_negatives:
                neg_id = np.random.choice(vocab_size, p=vocab.noise_dist)
                if neg_id != context_id and neg_id != center_id:
                    neg_ids.append(neg_id)
            pairs.append((center_id, context_id, neg_ids))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{corpus_length - 2*window_size} tokens", end="\r")

    save_training_pairs(pairs, pairs_filename)

    return pairs
