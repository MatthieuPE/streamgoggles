"""Sample dataclass: one labeled training/eval sample.

A Sample bundles map, label, valid_mask, parameters, and metadata:
- map_stack: (n_dist, ny, nx) projected map stack (float32).
- label_stack: (n_dist, ny, nx) target labels (float32).
- valid_mask: (ny, nx) bool mask of valid pixels.
- params: dict of stream parameters used to generate this sample.
- metadata: optional dict with version info, timings, etc.

Rationale: Encapsulate a single sample so datasets and storage can work
with a standard container.
"""

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class Sample:
    """One labeled training/eval sample.
    
    Attributes:
        map_stack: (n_dist, ny, nx) float32 array of projected count maps.
            One channel per trial distance modulus.
        
        label_stack: (n_dist, ny, nx) float32 array of target labels.
            Policy-dependent: binary mask, density map, or soft-distance weights.
            One channel per trial distance modulus.
        
        valid_mask: (ny, nx) bool array. True where pixel has valid data
            (inside footprint), False where invalid (outside or no coverage).
            Same valid_mask applied to all channels of map_stack and label_stack.
        
        params: dict mapping parameter name -> value.
            E.g., {morphology: "uniform", width: 0.2, distance_modulus: 17.5, ...}.
        
        metadata: optional dict with additional info.
            Suggested keys: streamobs_version, code_version, seed, created_at, etc.
    
    Rationale: Bundles all data for one sample so it can be stored/loaded atomically,
    passed through datasets unchanged, and reconstructed from cache deterministically.
    """
    map_stack: np.ndarray
    label_stack: np.ndarray
    valid_mask: np.ndarray
    params: dict
    metadata: dict | None = None
    
    def __post_init__(self):
        """Validate shapes and types."""
        if self.map_stack.ndim != 3:
            raise ValueError(
                f"map_stack must be 3D (n_dist, ny, nx), got shape {self.map_stack.shape}"
            )
        if self.label_stack.shape != self.map_stack.shape:
            raise ValueError(
                f"label_stack shape {self.label_stack.shape} != map_stack shape "
                f"{self.map_stack.shape}"
            )
        if self.valid_mask.ndim != 2:
            raise ValueError(
                f"valid_mask must be 2D (ny, nx), got shape {self.valid_mask.shape}"
            )
        if self.valid_mask.shape != self.map_stack.shape[1:]:
            raise ValueError(
                f"valid_mask shape {self.valid_mask.shape} doesn't match "
                f"map_stack spatial dims {self.map_stack.shape[1:]}"
            )
        if self.map_stack.dtype != np.float32:
            raise ValueError(f"map_stack must be float32, got {self.map_stack.dtype}")
        if self.label_stack.dtype != np.float32:
            raise ValueError(f"label_stack must be float32, got {self.label_stack.dtype}")
        if self.valid_mask.dtype != bool:
            raise ValueError(f"valid_mask must be bool, got {self.valid_mask.dtype}")
    
    def is_pure_background(self) -> bool:
        """Check if this sample has no stream (label_stack all zeros where valid).
        
        Returns:
            True if label_stack is effectively zero everywhere a stream is present.
        
        Rationale: Useful for statistics (what fraction of training is backgrounds?).
        """
        raise NotImplementedError
    
    @property
    def n_channels(self) -> int:
        """Number of distance channels (distance moduli)."""
        return self.map_stack.shape[0]
    
    @property
    def image_size(self) -> tuple[int, int]:
        """Spatial size (height, width) of map and label."""
        return self.map_stack.shape[1:]
