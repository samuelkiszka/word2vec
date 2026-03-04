"""
utils.py
========
Evaluation helpers: cosine similarity, nearest neighbors, analogy tests.
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

def analogy_test(
    word_a: str,
    word_b: str,
    word_c: str,
    vocab: Vocabulary,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k words that best complete the analogy 'a is to b as c is to ?'."""
    for w in [word_a, word_b, word_c]:
        if w not in vocab.word2idx:
            raise ValueError(f"Word '{w}' not in vocabulary.")

    idx_a = vocab.word2idx[word_a]
    idx_b = vocab.word2idx[word_b]
    idx_c = vocab.word2idx[word_c]

    vec_a = embeddings[idx_a]
    vec_b = embeddings[idx_b]
    vec_c = embeddings[idx_c]

    # Compute the target vector for the analogy
    target_vec = vec_b - vec_a + vec_c

    similarities = []
    for i, other_vec in enumerate(embeddings):
        if i in {idx_a, idx_b, idx_c}:
            continue  # skip the words in the analogy
        sim = cosine_similarity(target_vec, other_vec)
        similarities.append((vocab.idx2word[i], sim))

    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
