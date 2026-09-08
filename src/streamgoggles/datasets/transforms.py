"""Data transforms: normalization, augmentation, valid_mask handling.

Rationale: Isolate preprocessing (normalization, augmentation) from dataset logic.
Ensure all transforms respect valid_mask (don't corrupt/propagate invalid pixels).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RobustNormalizer:
    """Per-channel robust normalization (computed over valid pixels only).
    
    Computes mean/std per channel from clean (valid) data and normalizes
    new data: (x - mean) / std. Applied independently per channel.
    
    Rationale: Robust normalization (ignoring NaNs/invalid pixels) prevents
    edge artifacts. Per-channel prevents one bright channel from dominating.
    """
    
    def __init__(self):
        """Initialize normalizer (not yet fitted)."""
        self.mean = None
        self.std = None
    
    def fit(self, map_stack: np.ndarray, valid_mask: np.ndarray) -> None:
        """Compute mean/std per channel over valid (clean) pixels.
        
        Parameters:
            map_stack: (n_dist, ny, nx) array.
            valid_mask: (ny, nx) bool mask; True for valid pixels.
        
        Rationale: Fits on a subset of training data (or full training set)
        to get robust statistics. Should be called once during dataloader
        initialization.
        """
        raise NotImplementedError
    
    def __call__(self, map_stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Normalize map stack in-place (or copy).
        
        Parameters:
            map_stack: (n_dist, ny, nx) array.
            valid_mask: (ny, nx) bool mask.
        
        Returns:
            Normalized map_stack, same shape. Invalid pixels untouched.
        
        Raises:
            RuntimeError if normalizer not yet fitted.
        """
        raise NotImplementedError


class StreamMapTransform:
    """Torch-compatible transform: normalization, augmentation, noise.
    
    Attributes:
        normalizer: RobustNormalizer instance (optional).
        augment: bool, apply random augmentation (flips, rotations).
        noise_std: float, add Gaussian noise with this std (before normalization).
    
    Rationale: Single transform handles all preprocessing. Can be stacked
    with torchvision.transforms.Compose if needed. Respects valid_mask
    throughout (augmentation is identity when mask is masked).
    """
    
    def __init__(self, normalizer=None, augment: bool = False, noise_std: float = 0.0):
        """Initialize transform.
        
        Parameters:
            normalizer: RobustNormalizer instance (if None, no normalization).
            augment: If True, apply random flips/90° rotations.
            noise_std: Std of Gaussian noise added to valid pixels (0 = no noise).
        """
        self.normalizer = normalizer
        self.augment = augment
        self.noise_std = noise_std
    
    def __call__(self, sample: dict) -> dict:
        """Apply transform to sample dict.
        
        Parameters:
            sample: dict with map_stack, label_stack, valid_mask.
        
        Returns:
            dict with transformed arrays (same keys).
        
        Raises:
            KeyError if required keys missing from sample.
        """
        raise NotImplementedError
