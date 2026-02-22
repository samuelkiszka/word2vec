"""
model.py
========
Holds the two embedding matrices and the forward pass.

Architecture:
  W_in  : shape (vocab_size, emb_dim)  — centre-word ("input") embeddings
  W_out : shape (vocab_size, emb_dim)  — context-word ("output") embeddings

After training, W_in is used as word vectors.
"""

import numpy as np


class Word2Vec:
    def __init__(self, vocab_size: int, emb_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        emb_dim = emb_dim
        # Initialise small random values (common: uniform in [-0.5/d, 0.5/d])
        self.W_in  = rng.uniform(-0.5 / emb_dim, 0.5 / emb_dim, (vocab_size, emb_dim))
        self.W_out = np.zeros((vocab_size, emb_dim))

    def forward(
        self,
        center_id: int,
        context_id: int,
        neg_ids: list[int],
    ) -> tuple[float, np.ndarray]:
        """
        Returns (score_pos, score_neg_array).
        """
        v_w = self.W_in[center_id]          # (emb_dim,)
        v_c = self.W_out[context_id]        # (emb_dim,)
        v_neg = self.W_out[neg_ids]         # (K, emb_dim)

        score_pos = v_c @ v_w               # scalar
        score_neg = v_neg @ v_w             # (K,)

        return score_pos, score_neg

    def get_embedding(self, word_idx: int) -> np.ndarray:
        return self.W_in[word_idx]

    def load_embeddings(self, W_in: np.ndarray):
        self.W_in = W_in


