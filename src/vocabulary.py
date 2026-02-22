"""
vocabulary.py
=============
Handles vocabulary construction, and word subsampling.
"""

from collections import Counter
import numpy as np


class Vocabulary:
    def __init__(self, min_count: int = 5, subsample_t: float = 1e-5):
        self.min_count = min_count
        self.subsample_t = subsample_t

        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.word_freqs: np.ndarray = None   # raw frequency counts
        self.noise_dist: np.ndarray = None   # P_n(w) ∝ freq^(3/4)

    def build(self, tokens: list[str]) -> None:
        """Build vocab from a flat list of tokens."""
        # Count word frequencies
        counter = Counter(tokens)
        # Filter out low-frequency words
        filtered = {w: c for w, c in counter.items() if c >= self.min_count}
        # Sort by frequency (descending) and then alphabetically (ascending)
        sorted_words = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))

        # Build word2idx and idx2word
        self.idx2word = [w for w, _ in sorted_words]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

        # Store raw frequency counts in the same order as idx2word
        self.word_freqs = np.array([filtered[w] for w in self.idx2word], dtype=np.float64)

        # Compute noise distribution
        freq_pow = np.power(self.word_freqs, 0.75)
        self.noise_dist = freq_pow / np.sum(freq_pow)


    def subsample(self, tokens: list[str]) -> list[str]:
        """Drop frequent tokens stochastically."""
        if self.subsample_t <= 0:
            return tokens  # no subsampling

        total_count = np.sum(self.word_freqs)
        freq_ratios = self.word_freqs / total_count
        discard_probs = 1 - np.sqrt(self.subsample_t / freq_ratios)

        # Map word to discard probability
        word_discard_prob = {w: discard_probs[i] for i, w in enumerate(self.idx2word)}

        # Subsample tokens
        subsampled = []
        for t in tokens:
            if t in word_discard_prob:
                if np.random.rand() >= word_discard_prob[t]:
                    subsampled.append(t)
            else:
                subsampled.append(t)  # keep OOV words (if any)

        return subsampled

    def __len__(self) -> int:
        return len(self.idx2word)
