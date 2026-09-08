"""Segmentation metrics (IoU, Dice, precision, recall, MSE, etc).

All metrics accept:
- pred: model predictions or binary mask.
- target: ground-truth labels.
- valid_mask: optional bool mask of valid pixels.

Metrics are computed excluding invalid pixels (not downweighted, excluded).

Rationale: Standard metrics for segmentation and regression, all respecting
the valid_mask convention.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def iou(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5
) -> float:
    """Intersection over Union (binary metric).
    
    IoU = |pred ∩ target| / |pred ∪ target|.
    Computed over valid pixels only.
    
    Parameters:
        pred: predictions (float), shape (ny, nx) or (n_channels, ny, nx).
        target: ground truth (float), shape same as pred.
        valid_mask: optional (ny, nx) bool mask.
        threshold: threshold for binarizing pred (default 0.5).
    
    Returns:
        IoU value in [0, 1].
    
    Raises:
        ValueError if shapes incompatible.
    """
    raise NotImplementedError


def dice(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5
) -> float:
    """Dice coefficient (soft Dice).
    
    Dice = 2 * |pred ∩ target| / (|pred| + |target|).
    
    Parameters:
        pred, target, valid_mask, threshold: as in iou().
    
    Returns:
        Dice value in [0, 1].
    """
    raise NotImplementedError


def precision_recall(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5
) -> tuple[float, float]:
    """Binary precision and recall.
    
    Precision = TP / (TP + FP).
    Recall = TP / (TP + FN).
    
    Parameters:
        pred, target, valid_mask, threshold: as in iou().
    
    Returns:
        (precision, recall) tuple.
    """
    raise NotImplementedError


def mse(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None
) -> float:
    """Mean squared error (regression metric).
    
    MSE = mean((pred - target)^2) over valid pixels.
    
    Parameters:
        pred, target: same shapes, float.
        valid_mask: optional (ny, nx) bool mask.
    
    Returns:
        MSE value (float, ≥ 0).
    """
    raise NotImplementedError


def correlation(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None
) -> float:
    """Pearson correlation coefficient.
    
    Correlation = cov(pred, target) / (std(pred) * std(target)).
    Computed over valid pixels only.
    
    Parameters:
        pred, target: same shapes, float.
        valid_mask: optional (ny, nx) bool mask.
    
    Returns:
        Correlation in [-1, 1].
    """
    raise NotImplementedError


def weighted_recall(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.5
) -> float:
    """Recall weighted by target (density mode metric).
    
    Emphasizes recovery of high-density regions.
    weighted_recall = sum(target * TP) / sum(target * (TP + FN)).
    
    Parameters:
        pred, target, valid_mask, threshold: as in iou().
    
    Returns:
        Weighted recall in [0, 1].
    """
    raise NotImplementedError
