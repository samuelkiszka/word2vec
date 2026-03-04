"""
gradients.py
============
Manual backward pass for the negative-sampling loss.

Let:
  u = v_w  (center embedding / context mean)
  v = v_c  (positive context embedding / target embedding)
  n_k      (negative context embeddings)

Loss:
  L = -log sigmoid(v*u) - Σ_k log σ(-n_k*u)

Gradients w.r.t. output embeddings:
  dL/dv    = -(1 - σ(v*u))  * u       # gradient for positive context row
  dL/dn_k  =  (1 - σ(-n_k*u)) * u     # = σ(n_k*u) * u  for each negative

Gradient w.r.t. input embedding:
  dL/du = -(1 - σ(v*u)) * v  +  Σ_k σ(n_k*u) * n_k

Key insight: only 1 + K rows of W_out and 1 (or 2 * win_size for cbowns) row of W_in are touched per sample.
"""

import numpy as np

def compute_gradients(
    v_w: np.ndarray,        # centre embedding / context mean  (emb_dim,)
    v_c: np.ndarray,        # positive context / target embedding (emb_dim,)
    v_neg: np.ndarray,      # negative contexts (K, emb_dim)
    sig_pos: float,
    sig_neg: np.ndarray,  # (K,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      grad_v_w   — gradient w.r.t. centre embedding   (emb_dim,)
      grad_v_c   — gradient w.r.t. positive context   (emb_dim,)
      grad_v_neg — gradient w.r.t. negative contexts  (K, emb_dim)
    """
    # Gradients w.r.t. output embeddings
    grad_v_c = -(1 - sig_pos) * v_w             # (emb_dim,)
    grad_v_neg = sig_neg[:, np.newaxis] * v_w   # (K, emb_dim)

    # Gradient w.r.t. input embedding
    grad_v_w = (
            -(1 - sig_pos) * v_c
            + np.sum(sig_neg[:, np.newaxis] * v_neg, axis=0)
    )  # (emb_dim,)

    return grad_v_w, grad_v_c, grad_v_neg