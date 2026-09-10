"""Segmentation metrics (IoU, Dice, precision, recall, MSE, etc).

All metrics accept:
- pred: model predictions or binary mask.
- target: ground-truth labels.
- valid_mask: optional bool mask of valid pixels.

Metrics are computed excluding invalid pixels (not downweighted, excluded).

Undefined-value convention: any metric with an undefined ratio (e.g. IoU
with nothing predicted and nothing true, precision with no positive
predictions, correlation with too few valid pixels) returns `float("nan")`
rather than a guessed 0.0/1.0 -- so it never silently reads as a real
score. Callers aggregating over many samples (evaluation/completeness_purity.py)
should `.dropna()`/ignore NaNs explicitly rather than have this module pick
a value for them.

Threshold-based metrics (iou/dice/precision_recall/weighted_recall) work on
`pred > threshold` and `target > threshold`, whatever scale `pred`/`target`
are actually in -- a probability in [0, 1] for the binary/density
label_policy options, or a literal star count for the current default
label_policy="stream_count" (injector.py) -- so `threshold` should be
chosen accordingly by the caller (e.g. threshold=0.5 stars is a reasonable
"is there a true member here" cut for stream_count's integer target).

Rationale: Standard metrics for segmentation and regression, all respecting
the valid_mask convention.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _check_shapes(
    pred: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim not in (2, 3):
        raise ValueError(
            f"pred must be (ny, nx) or (n_channels, ny, nx), got shape {pred.shape}"
        )
    return pred, target


def _broadcast_valid_mask(
    pred: np.ndarray, valid_mask: np.ndarray | None
) -> np.ndarray:
    """Broadcast valid_mask to pred's exact shape.

    valid_mask is spatial-only ((ny, nx), shared across channels) or
    already matches pred's full shape exactly.
    """
    if valid_mask is None:
        return np.ones(pred.shape, dtype=bool)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape == pred.shape:
        return valid_mask
    if pred.ndim == 3 and valid_mask.shape == pred.shape[-2:]:
        return np.broadcast_to(valid_mask, pred.shape)
    raise ValueError(
        f"valid_mask shape {valid_mask.shape} doesn't match pred shape {pred.shape} "
        f"(expected {pred.shape} or, for 3D pred, {pred.shape[-2:]})"
    )


def iou(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    """Intersection over Union (binary metric).

    IoU = |pred ∩ target| / |pred ∪ target|.
    Computed over valid pixels only.

    Parameters:
        pred: predictions (float), shape (ny, nx) or (n_channels, ny, nx).
        target: ground truth (float), shape same as pred.
        valid_mask: optional (ny, nx) bool mask.
        threshold: threshold for binarizing pred and target (default 0.5).

    Returns:
        IoU value in [0, 1], or NaN if the union is empty (nothing
        predicted and nothing true within the valid region).

    Raises:
        ValueError if shapes incompatible.
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)

    pred_bin = (pred > threshold) & mask
    target_bin = (target > threshold) & mask

    union = np.count_nonzero(pred_bin | target_bin)
    if union == 0:
        return float("nan")
    intersection = np.count_nonzero(pred_bin & target_bin)
    return intersection / union


def dice(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    """Dice coefficient.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|), on `pred`/`target`
    binarized at `threshold` (matching iou()'s convention exactly, despite
    the "soft" label in some descriptions of this formula elsewhere --
    unlike models.losses.DiceLoss, which stays continuous, this evaluation
    metric binarizes for an interpretable detection statistic).

    Parameters:
        pred, target, valid_mask, threshold: as in iou().

    Returns:
        Dice value in [0, 1], or NaN if both |pred| and |target| are 0
        within the valid region.
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)

    pred_bin = (pred > threshold) & mask
    target_bin = (target > threshold) & mask

    denom = np.count_nonzero(pred_bin) + np.count_nonzero(target_bin)
    if denom == 0:
        return float("nan")
    intersection = np.count_nonzero(pred_bin & target_bin)
    return 2.0 * intersection / denom


def precision_recall(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Binary precision and recall.

    Precision = TP / (TP + FP).
    Recall = TP / (TP + FN).

    Parameters:
        pred, target, valid_mask, threshold: as in iou().

    Returns:
        (precision, recall) tuple, each in [0, 1] or NaN if its own
        denominator is 0 (no positive predictions for precision, no true
        positives at all for recall).
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)

    pred_bin = (pred > threshold) & mask
    target_bin = (target > threshold) & mask

    tp = np.count_nonzero(pred_bin & target_bin)
    fp = np.count_nonzero(pred_bin & ~target_bin)
    fn = np.count_nonzero(~pred_bin & target_bin)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def mse(
    pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray | None = None
) -> float:
    """Mean squared error (regression metric).

    MSE = mean((pred - target)^2) over valid pixels.

    Parameters:
        pred, target: same shapes, float.
        valid_mask: optional (ny, nx) bool mask.

    Returns:
        MSE value (float, >= 0), or NaN if there are no valid pixels.
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)
    if not mask.any():
        return float("nan")
    return float(np.mean((pred[mask] - target[mask]) ** 2))


def correlation(
    pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray | None = None
) -> float:
    """Pearson correlation coefficient.

    Correlation = cov(pred, target) / (std(pred) * std(target)).
    Computed over valid pixels only.

    Parameters:
        pred, target: same shapes, float.
        valid_mask: optional (ny, nx) bool mask.

    Returns:
        Correlation in [-1, 1], or NaN if fewer than 2 valid pixels or
        either array is constant over the valid region (zero variance).
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)

    pred_valid = pred[mask]
    target_valid = target[mask]
    if pred_valid.size < 2 or np.std(pred_valid) == 0 or np.std(target_valid) == 0:
        return float("nan")
    return float(np.corrcoef(pred_valid, target_valid)[0, 1])


def weighted_recall(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    """Recall weighted by target (density/count mode metric).

    Emphasizes recovery of high-density (or, under label_policy=
    "stream_count", high-count) regions over low ones: missing a pixel
    with many true stream stars costs more than missing a pixel with one.

    weighted_recall = sum(target * TP) / sum(target * (TP + FN))
                     = sum(target over correctly-flagged true pixels)
                       / sum(target over all true pixels)
    (TP + FN, per true pixel, is just "this pixel is a true pixel", so the
    denominator reduces to the total target mass among target_bin pixels.)

    Parameters:
        pred, target, valid_mask, threshold: as in iou().

    Returns:
        Weighted recall in [0, 1], or NaN if there's no true target mass
        to recover (target <= threshold everywhere in the valid region).
    """
    pred, target = _check_shapes(pred, target)
    mask = _broadcast_valid_mask(pred, valid_mask)

    pred_bin = (pred > threshold) & mask
    target_bin = (target > threshold) & mask

    denominator = target[target_bin].sum()
    if denominator == 0:
        return float("nan")
    numerator = target[pred_bin & target_bin].sum()
    return float(numerator / denominator)
