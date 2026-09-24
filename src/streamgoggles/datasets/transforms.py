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


def _worker_local_rng(owner) -> None:
    """Give each `DataLoader` worker its own stream of ``owner.rng``.

    Workers each receive a copy of the transform with ``rng`` in exactly the
    parent's state, so without this every worker draws the same sequence of
    augmentations (or queried distances) in lockstep. Each worker takes its
    own child of the shared parent, deterministically, as
    `StreamMapDataset` does for its samples.
    """
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return
    info = get_worker_info()
    if info is None or getattr(owner, "_worker_id", None) == info.id:
        return
    owner.rng = owner.rng.spawn(info.num_workers)[info.id]
    owner._worker_id = info.id


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


class WindowNormalizer:
    r"""Standardize each channel of a window from that window alone.

    For one window and one channel ``c``, with ``V`` the window's valid pixels
    (those whose HEALPix neighbours all lie inside the footprint):

    .. math::

        \mu_c = \frac{1}{|V|} \sum_{p \in V} x_c(p), \qquad
        \sigma_c = \sqrt{\frac{1}{|V|} \sum_{p \in V} \big(x_c(p) - \mu_c\big)^2},
        \qquad
        x'_c(p) = \frac{x_c(p) - \mu_c}{\sigma_c} \quad (p \in V).

    Invalid pixels keep their fill value (0), and a channel that is constant
    over ``V`` (``sigma_c = 0``) is only centred. Nothing is fitted or stored:
    unlike `RobustNormalizer`, whose mean and standard deviation per channel
    are estimated once on training windows and then applied to every window,
    this uses no statistic from any other window or from the background. On
    real data the background level is not known in advance, so the input must
    not depend on assuming one. It follows that the absolute density of the
    sky is removed: a window and the same window with every count doubled give
    the same input.

    Called like `RobustNormalizer` (``normalizer(map_stack, valid_mask)``), so
    it plugs into `StreamMapTransform` unchanged.
    """

    def __call__(self, map_stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Normalized copy of ``map_stack`` (n_channels, ny, nx)."""
        if valid_mask.shape != map_stack.shape[1:]:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} doesn't match map_stack "
                f"spatial dims {map_stack.shape[1:]}"
            )
        out = map_stack.astype(np.float32, copy=True)
        if not valid_mask.any():
            return out
        for c in range(map_stack.shape[0]):
            values = map_stack[c][valid_mask].astype(np.float64)
            mean = values.mean()
            std = values.std()
            out[c][valid_mask] = (values - mean) / (std if std > 0 else 1.0)
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
        if self.augment:
            _worker_local_rng(self)

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


class QueryDistanceTransform:
    """The model's input at one queried trial distance.

    Windows are built with one channel per (trial distance, filter), in the
    order of ``metadata["channels"]`` (each ``{"filter", "distance_modulus"}``).
    The model does not see them all. For a queried distance modulus ``dm`` it
    gets, in this order:

    - channel 0: the matched filter's map at ``dm - step``;
    - channel 1: the matched filter's map at ``dm``;
    - channel 2: the matched filter's map at ``dm + step``;
    - channel 3: the decoy's map (the fixed colour-magnitude box does not
      depend on the trial distance, so the one at ``dm`` stands for all);
    - channels 4-6: constant maps holding the distance moduli of channels 0-2,
      scaled as ``(dm - center) / scale`` so the same distance always reads
      the same.

    It learns the matched filter's label at ``dm`` alone: ``label_stack``
    becomes that one channel. The neighbouring maps let it use the stream's
    signal at nearby distances -- for a distance gradient, for instance --
    while it answers for one distance.

    ``inner`` (a `StreamMapTransform`) is applied to the full stack first, so
    each (distance, filter) map is normalized with its own statistics, whatever
    query position it then lands in, and the constant maps are added after it:
    distances are never normalized per sample.

    Which distance is queried: a fixed ``query`` (evaluation, and inference,
    which slides it over ``query_grid``), or, when ``query`` is None, one drawn
    per sample (training): the grid point nearest the stream's true distance,
    shifted by ``k * step`` with ``k`` uniform in ``-max_shift..max_shift``,
    among the points that fall inside ``query_grid``. A stream-free window
    (no ``distance_modulus`` in its params) gets a uniform draw on the grid.
    The queried value is recorded in ``params["query_distance_modulus"]``.

    Attributes:
        inner: `StreamMapTransform` for the full channel stack (normalization,
            augmentation).
        query_grid: trial distances that can be queried; each needs its
            neighbours at ``+- step`` among the window's channels.
        step: distance between the three matched-filter maps, mag.
        matched_filter, decoy: filter names in ``metadata["channels"]``.
        query: fixed queried distance, or None to draw one per sample.
        max_shift: largest shift from the true distance in training, in steps.
        center, scale: the constant maps hold ``(dm - center) / scale``.
        rng: draws the training query.
    """

    n_channels = 7

    def __init__(
        self,
        inner,
        query_grid,
        step: float = 0.5,
        matched_filter: str = "good",
        decoy: str = "decoy",
        query: float | None = None,
        max_shift: int = 2,
        center: float = 17.0,
        scale: float = 2.0,
        rng: np.random.Generator | None = None,
    ):
        self.inner = inner
        self.query_grid = np.asarray(sorted(query_grid), dtype=float)
        self.step = float(step)
        self.matched_filter = matched_filter
        self.decoy = decoy
        self.query = query
        self.max_shift = int(max_shift)
        self.center = float(center)
        self.scale = float(scale)
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_query(self, params: dict) -> float:
        """The distance to query for a sample with these parameters."""
        if self.query is not None:
            return float(self.query)
        _worker_local_rng(self)
        true = params.get("distance_modulus") if params else None
        if true is None:
            return float(self.rng.choice(self.query_grid))
        nearest = self.query_grid[np.argmin(np.abs(self.query_grid - float(true)))]
        shifts = nearest + self.step * np.arange(-self.max_shift, self.max_shift + 1)
        allowed = [
            q for q in shifts if np.any(np.isclose(self.query_grid, q, atol=1e-6))
        ]
        return float(self.rng.choice(allowed))

    @staticmethod
    def channel_index(channels, name: str, dm: float) -> int:
        """Index of the (filter, distance) channel, or KeyError."""
        for index, channel in enumerate(channels):
            if channel["filter"] == name and np.isclose(
                float(channel["distance_modulus"]), dm, atol=1e-6
            ):
                return index
        raise KeyError(f"no channel for filter {name!r} at distance modulus {dm:g}")

    def __call__(self, sample: dict) -> dict:
        """Apply `inner`, then keep the query's channels and add its distances."""
        channels = sample["metadata"]["channels"]
        params = sample.get("params") or {}
        dm = self.choose_query(params)
        distances = (dm - self.step, dm, dm + self.step)
        maps = [self.channel_index(channels, self.matched_filter, d) for d in distances]
        decoy = self.channel_index(channels, self.decoy, dm)

        out = self.inner(sample)
        stack = out["map_stack"]
        constant = [
            np.full(stack.shape[1:], (d - self.center) / self.scale, dtype=np.float32)
            for d in distances
        ]
        out["map_stack"] = np.ascontiguousarray(
            np.concatenate([stack[maps + [decoy]], np.stack(constant)]),
            dtype=np.float32,
        )
        out["label_stack"] = np.ascontiguousarray(out["label_stack"][[maps[1]]])
        out["params"] = {**params, "query_distance_modulus": dm}
        return out
