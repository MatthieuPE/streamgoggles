"""Data transforms: normalization, augmentation, valid_mask handling.

Rationale: Isolate preprocessing (normalization, augmentation) from dataset logic.
Ensure all transforms respect valid_mask (don't corrupt/propagate invalid pixels):
per-channel normalization is computed over valid pixels only and leaves invalid
pixels untouched; flips/90-degree rotations are applied identically to map_stack,
label_stack, AND valid_mask, so they stay in geometric lockstep.

No synthetic noise injection: map_stack is count data (Poisson-like, especially
at low counts), already carries the survey's own realistic photometric noise
from injection, and the label is now the true star count (2026-09-09 pivot) --
adding uncorrelated Gaussian noise on top would teach the network a noise
model that doesn't match the real one, which matters more for a counting/
denoising target than it would for classification. Considered and removed.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RobustNormalizer:
    """Per-channel robust normalization (computed over valid pixels only).

    Computes mean/std per channel from clean (valid) data and normalizes
    new data: (x - mean) / std. Applied independently per channel.

    Rationale: Robust normalization (ignoring invalid pixels, which are a
    fixed 0.0 fill value -- decision 22 -- not real data) prevents them from
    skewing the statistics or getting rescaled as if they were. Per-channel
    prevents one bright channel (e.g. a "good" filter's map, much richer
    than a "decoy" filter's at the same distance) from dominating another's
    scale.
    """

    def __init__(self):
        """Initialize normalizer (not yet fitted)."""
        self.mean = None
        self.std = None

    def fit(self, map_stack: np.ndarray, valid_mask: np.ndarray) -> None:
        """Compute mean/std per channel over valid (clean) pixels.

        Parameters:
            map_stack: (n_channels, ny, nx) array. May be a single sample's
                map_stack or several concatenated along a new leading axis
                and reshaped to (n_channels, -1) beforehand, for statistics
                pooled across multiple samples -- this method itself only
                ever sees one (map_stack, valid_mask) pair, so pooling
                across samples is the caller's responsibility.
            valid_mask: (ny, nx) bool mask; True for valid pixels, shared by
                every channel of `map_stack` (matching sample.Sample's own
                convention).

        Raises:
            ValueError if shapes are inconsistent, or if a channel has no
                valid pixels to compute statistics from.

        Rationale: Fits on a subset of training data (or full training set)
        to get robust statistics. Should be called once during dataloader
        initialization.
        """
        if map_stack.ndim != 3:
            raise ValueError(
                f"map_stack must be 3D (n_channels, ny, nx), got shape {map_stack.shape}"
            )
        if valid_mask.shape != map_stack.shape[1:]:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} doesn't match map_stack "
                f"spatial dims {map_stack.shape[1:]}"
            )

        n_channels = map_stack.shape[0]
        mean = np.zeros(n_channels, dtype=np.float64)
        std = np.zeros(n_channels, dtype=np.float64)
        for c in range(n_channels):
            values = map_stack[c][valid_mask]
            if values.size == 0:
                raise ValueError(f"channel {c} has no valid pixels to fit on")
            mean[c] = values.mean()
            channel_std = values.std()
            # A constant-valued channel (std=0) would divide by zero;
            # leaving it unscaled (std=1) is the only sane fallback.
            std[c] = channel_std if channel_std > 0 else 1.0

        self.mean = mean
        self.std = std

    def __call__(self, map_stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Normalize map stack, leaving invalid pixels unchanged.

        Parameters:
            map_stack: (n_channels, ny, nx) array.
            valid_mask: (ny, nx) bool mask.

        Returns:
            Normalized map_stack, same shape (a new array; the input is not
            modified). Invalid pixels untouched (kept at their original
            fill value, not shifted/rescaled).

        Raises:
            RuntimeError if normalizer not yet fitted.
            ValueError if `map_stack`'s channel count doesn't match what
                `fit()` was called with.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("RobustNormalizer must be fit() before use")
        if map_stack.shape[0] != len(self.mean):
            raise ValueError(
                f"map_stack has {map_stack.shape[0]} channels, normalizer was "
                f"fit on {len(self.mean)}"
            )

        out = map_stack.astype(np.float32, copy=True)
        for c in range(map_stack.shape[0]):
            out[c][valid_mask] = (map_stack[c][valid_mask] - self.mean[c]) / self.std[c]
        return out


class StreamMapTransform:
    """Torch-compatible transform: normalization + augmentation.

    Attributes:
        normalizer: RobustNormalizer instance (optional; applied to
            map_stack only -- label_stack stays in its raw count unit).
        augment: bool, apply random augmentation (flips, 90-degree rotations).
        rng: np.random.Generator driving augmentation randomness (not in the
            original skeleton sketch -- added for the same reproducibility
            reason every other stochastic component in this project takes an
            explicit rng rather than using global numpy state; defaults to a
            fresh `np.random.default_rng()`).

    Rationale: Single transform handles all preprocessing. Can be stacked
    with torchvision.transforms.Compose if needed. Respects valid_mask
    throughout: flips/rotations are applied identically to map_stack,
    label_stack, AND valid_mask, so all three stay in geometric lockstep
    after augmentation.
    """

    def __init__(
        self,
        normalizer: RobustNormalizer | None = None,
        augment: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """Initialize transform.

        Parameters:
            normalizer: RobustNormalizer instance (if None, no normalization).
            augment: If True, apply random flips/90-degree rotations.
            rng: np.random.Generator (if None, a default one is created).
        """
        self.normalizer = normalizer
        self.augment = augment
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, sample: dict) -> dict:
        """Apply transform to sample dict.

        Parameters:
            sample: dict with map_stack, label_stack, valid_mask (any other
                keys, e.g. params/metadata, are passed through unchanged).

        Returns:
            dict with transformed arrays (same keys as `sample`; map_stack/
            label_stack float32, valid_mask bool -- new arrays, `sample`
            itself is not modified).

        Raises:
            KeyError if required keys missing from sample.
        """
        for key in ("map_stack", "label_stack", "valid_mask"):
            if key not in sample:
                raise KeyError(f"sample is missing required key {key!r}")

        map_stack = sample["map_stack"].copy()
        label_stack = sample["label_stack"].copy()
        valid_mask = sample["valid_mask"].copy()

        if self.augment:
            k = int(self.rng.integers(0, 4))
            map_stack = np.rot90(map_stack, k, axes=(-2, -1))
            label_stack = np.rot90(label_stack, k, axes=(-2, -1))
            valid_mask = np.rot90(valid_mask, k, axes=(-2, -1))

            if self.rng.random() < 0.5:
                map_stack = np.flip(map_stack, axis=-1)
                label_stack = np.flip(label_stack, axis=-1)
                valid_mask = np.flip(valid_mask, axis=-1)
            if self.rng.random() < 0.5:
                map_stack = np.flip(map_stack, axis=-2)
                label_stack = np.flip(label_stack, axis=-2)
                valid_mask = np.flip(valid_mask, axis=-2)

            # rot90/flip return views with negative/non-standard strides,
            # which torch.from_numpy() rejects -- copy to plain contiguous
            # arrays before handing this off to any torch-facing code.
            map_stack = np.ascontiguousarray(map_stack)
            label_stack = np.ascontiguousarray(label_stack)
            valid_mask = np.ascontiguousarray(valid_mask)

        if self.normalizer is not None:
            map_stack = self.normalizer(map_stack, valid_mask)

        out = dict(sample)
        out["map_stack"] = map_stack.astype(np.float32, copy=False)
        out["label_stack"] = label_stack.astype(np.float32, copy=False)
        out["valid_mask"] = valid_mask.astype(bool, copy=False)
        return out
