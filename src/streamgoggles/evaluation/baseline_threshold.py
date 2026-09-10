"""Threshold baseline: k·σ on finalized matched-filter map.

Simple, non-learned baseline for comparison with network predictions.
Useful for validating that network learns something beyond a trivial threshold.

Rationale: Establishes upper bound on what thresholding alone can achieve,
making network improvements interpretable.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_baseline(
    finalized_map: np.ndarray, valid_mask: np.ndarray, k: float = 3.0
) -> np.ndarray:
    """Build binary prediction via k·σ thresholding on finalized map.

    Threshold = mean + k * std, computed over valid_mask==True pixels.
    Baseline predicts 1 where counts > threshold, 0 elsewhere (invalid pixels → 0).

    Parameters:
        finalized_map: full-sky HEALPix or projected 2D map (float), shape (npix,) or (ny, nx).
        valid_mask: bool mask, True for valid pixels. Same spatial shape as finalized_map.
        k: number of std above mean (default 3.0).

    Returns:
        Binary prediction map (0 or 1, dtype int), same shape as finalized_map.
        Always 0 at invalid pixels, regardless of their (meaningless) value
        in finalized_map.

    Raises:
        ValueError if finalized_map and valid_mask have incompatible shapes.
    """
    finalized_map = np.asarray(finalized_map, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if finalized_map.shape != valid_mask.shape:
        raise ValueError(
            f"finalized_map shape {finalized_map.shape} != valid_mask shape {valid_mask.shape}"
        )

    if not valid_mask.any():
        return np.zeros(finalized_map.shape, dtype=int)

    valid_values = finalized_map[valid_mask]
    threshold = valid_values.mean() + k * valid_values.std()

    prediction = np.zeros(finalized_map.shape, dtype=int)
    above_threshold = (finalized_map > threshold) & valid_mask
    prediction[above_threshold] = 1
    return prediction
