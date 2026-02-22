"""
loss.py
=======
Negative-sampling loss (binary cross-entropy form).

For one (center, context, negatives) triplet:

  L = - log σ(v_c * v_w)  -  Σ_k log σ(-v_{n_k} * v_w)

where σ(x) = 1 / (1 + exp(-x)).

This is equivalent to:
  L = - log σ(score_pos) - Σ_k log σ(-score_neg_k)
    = - log σ(score_pos) - Σ_k log (1 - σ(score_neg_k))
"""

import numpy as np


def ns_loss(sig_pos: float, sig_neg: np.ndarray) -> float:
    """
    Negative-sampling loss for one training sample.
    """
    # Compute the loss
    loss_pos = -np.log(sig_pos + 1e-10)                 # add small epsilon for stability
    loss_neg = -np.sum(np.log(1 - sig_neg + 1e-10))     # sum over negatives

    return loss_pos + loss_neg
