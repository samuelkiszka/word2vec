"""
trainer.py
==========
The main training loop — ties everything together.
"""

import numpy as np
from tqdm import tqdm
from src.model import Word2Vec
from src.loss import ns_loss
from src.gradients import compute_gradients
from src.utils import sigmoid


class Trainer:
    def __init__(
        self,
        model: Word2Vec,
        lr_start: float = 0.025,
        lr_min: float = 0.0001,
    ):
        self.model = model
        self.lr_start = lr_start
        self.lr_min = lr_min

    def train(
        self,
        training_pairs: list,   # [(center_id, context_id, [neg_ids]), ...]
        epochs: int = 5,
        save_file: str = "outputs/embeddings.npy",
    ) -> list[float]:
        """
        Run the full training loop.
        Returns list of per-epoch average losses.
        """
        total_steps = epochs * len(training_pairs)
        step = 0
        losses = []
        best_loss = float("inf")

        for epoch in range(epochs):
            np.random.shuffle(training_pairs)  # in-place shuffle
            epoch_loss = 0.0

            for center_id, context_id, neg_ids in tqdm(training_pairs, desc=f"Epoch {epoch+1}/{epochs}"):
                # Forward pass
                score_pos, score_neg = self.model.forward(center_id, context_id, neg_ids)
                # Compute sigmoid scores for loss and gradients
                sig_pos = sigmoid(np.array([score_pos]))[0] # scalar
                sig_neg = sigmoid(score_neg)                # (K,)
                # Compute loss
                loss = ns_loss(sig_pos, sig_neg)
                epoch_loss += loss
                # Compute gradients
                grad_w, grad_c, grad_neg = compute_gradients(
                    v_w=self.model.W_in[center_id],
                    v_c=self.model.W_out[context_id],
                    v_neg=self.model.W_out[neg_ids],
                    sig_pos=sig_pos,
                    sig_neg=sig_neg,
                )

                # Update embeddings (sparse updates)
                lr = max(self.lr_start * (1 - step / total_steps), self.lr_min)
                self.model.W_in[center_id]   -= lr * grad_w
                self.model.W_out[context_id] -= lr * grad_c
                self.model.W_out[neg_ids]    -= lr * grad_neg

                step += 1

            avg_loss = epoch_loss / len(training_pairs)
            losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_embeddings(save_file)

            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        return losses

    def save_embeddings(self, filename: str):
        np.save(filename, self.model.W_in)
        print(f"Embeddings saved to {filename}")
