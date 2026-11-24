"""Predictive-uncertainty decomposition for binary segmentation.

Given T stochastic forward passes (MC-Dropout samples, ensemble members or
bagged trees) of foreground probabilities p_t(x), the standard decomposition
(Kendall & Gal, 2017; Depeweg et al., 2018) is

    total      H[ mean_t p_t ]                 predictive entropy
    aleatoric  mean_t H[ p_t ]                 expected entropy  (data noise)
    epistemic  total - aleatoric  (>= 0)       mutual information (model)

All entropies are expressed in bits, so the maximum for a binary problem is 1.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def decompose(samples: np.ndarray) -> dict[str, np.ndarray]:
    """samples: (T, H, W) foreground probabilities."""
    samples = np.asarray(samples, np.float32)
    mean = samples.mean(0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(samples).mean(0)
    epistemic = np.clip(total - aleatoric, 0.0, None)
    return {"prob": mean, "total": total, "aleatoric": aleatoric, "epistemic": epistemic}


def review_score(total: np.ndarray, mask: np.ndarray, band: int = 3) -> float:
    """Image-level 'needs review' score: mean entropy in the predicted tumour and
    a thin band around it. Returns 0 when no tumour is predicted."""
    from scipy import ndimage as ndi

    if mask.sum() == 0:
        return 0.0
    region = ndi.binary_dilation(mask.astype(bool), iterations=band)
    return float(total[region].mean())
