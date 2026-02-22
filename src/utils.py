"""
utils.py
========
Evaluation helpers: cosine similarity, nearest neighbours, analogy tests.
"""

import numpy as np
from src.vocabulary import Vocabulary


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid:
          σ(x)  = 1/(1+e^{-x})  for x >= 0
                = e^x/(1+e^x)   for x < 0
    """
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out


def nearest_neighbours(
    word: str,
    vocab: Vocabulary,
    embeddings: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k most similar words by cosine similarity."""
    if word not in vocab.word2idx:
        raise ValueError(f"Word '{word}' not in vocabulary.")

    idx = vocab.word2idx[word]
    target_vec = embeddings[idx]

    similarities = []
    for i, other_vec in enumerate(embeddings):
        if i == idx:
            continue  # skip the query word itself
        sim = cosine_similarity(target_vec, other_vec)
        similarities.append((vocab.idx2word[i], sim))

    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
