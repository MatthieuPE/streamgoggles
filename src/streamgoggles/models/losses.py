"""Segmentation losses accepting soft targets and optional valid_mask.

All losses accept:
- pred: model output (B, C, H, W) or (C, H, W).
- target: target labels, same shape as pred, float or soft.
- valid_mask: optional (B, H, W) or (H, W) bool mask (matching pred's
  leading batch dim, if any); shared across every channel C, mirroring
  Sample.valid_mask's own convention (one mask, applied identically to
  every channel of map_stack/label_stack).

Masked reduction: compute loss only over valid_mask==True pixels.
This is NOT a class weight (we don't rebalance valid vs invalid);
it's a true EXCLUSION: invalid pixels contribute zero, not a weighted zero.

Rationale: Invalid pixels (outside survey footprint) should not affect training.
All losses must handle soft float targets [0, 1] for density labels.

Pred convention by loss (tied to UNet's `head` types, models/unet.py):
- DiceLoss/FocalLoss/TverskyLoss: pred is a probability in [0, 1]
  (UNet head="sigmoid").
- BCEWithLogitsLoss: pred is pre-sigmoid logits (UNet head="identity"),
  matching torch's own numerically-stable BCE-with-logits convention.
- MSELoss/WeightedMSELoss: pred is an unconstrained real-valued (or
  non-negative, UNet head="softplus") regression target -- the primary
  pair for this project's current default label_policy="stream_count"
  (injector.py), where target is a literal, non-negative star count, not
  a soft [0, 1] probability. dice/focal/tversky/bce were designed for
  and remain valid for the earlier binary/density label_policy options
  (rasterize.py, still selectable, not removed) -- see PLAN.md section 6.3.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _check_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
        )
    if pred.ndim not in (3, 4):
        raise ValueError(
            f"pred must be (B, C, H, W) or (C, H, W), got shape {tuple(pred.shape)}"
        )


def _broadcast_mask(
    pred: torch.Tensor, valid_mask: torch.Tensor | None
) -> torch.Tensor:
    """Return a bool mask broadcast to pred's exact shape.

    valid_mask is spatial-only (H, W) or batched-spatial (B, H, W) -- one
    mask shared across every channel -- so it gets a channel dim inserted
    (and, for the unbatched (C, H, W) pred case, no batch dim at all)
    before being expanded to match pred exactly.
    """
    if valid_mask is None:
        return torch.ones_like(pred, dtype=torch.bool)

    spatial_shape = tuple(pred.shape[-2:])
    if pred.ndim == 4:
        expected_unbatched = spatial_shape
        expected_batched = (pred.shape[0], *spatial_shape)
        if tuple(valid_mask.shape) == expected_unbatched:
            mask = valid_mask.view(1, 1, *spatial_shape)
        elif tuple(valid_mask.shape) == expected_batched:
            mask = valid_mask.view(pred.shape[0], 1, *spatial_shape)
        else:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} doesn't match pred "
                f"shape {tuple(pred.shape)} (expected {expected_unbatched} or "
                f"{expected_batched})"
            )
    else:
        if tuple(valid_mask.shape) != spatial_shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} doesn't match pred's "
                f"spatial shape {spatial_shape}"
            )
        mask = valid_mask.view(1, *spatial_shape)

    return mask.expand(pred.shape).to(torch.bool)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask==True elements; 0 if nothing is valid (no NaN)."""
    mask_f = mask.to(x.dtype)
    count = mask_f.sum().clamp(min=1.0)
    return (x * mask_f).sum() / count


def _spatial_sums(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum x*mask over the trailing (H, W) dims, keeping leading dims intact."""
    return (x * mask.to(x.dtype)).sum(dim=(-2, -1))


class MaskedLoss(nn.Module):
    """Base class for losses accepting valid_mask.

    Subclasses implement __call__(pred, target, valid_mask=None) and
    ensure that invalid pixels (valid_mask==False) contribute nothing
    to the loss.

    Rationale: Unified interface for all segmentation losses.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked loss.

        Parameters:
            pred: (B, C, H, W) or (C, H, W) model output.
            target: (B, C, H, W) or (C, H, W) target labels (float).
            valid_mask: optional (B, H, W) or (H, W) bool; True for valid pixels.

        Returns:
            Scalar loss value (reduction over batch and spatial dims,
            but only valid pixels).

        Raises:
            ValueError if shapes incompatible.
        """
        raise NotImplementedError


class DiceLoss(MaskedLoss):
    """Soft Dice loss, supports soft targets [0, 1].

    Dice = 2 * |pred ∩ target| / (|pred| + |target|).

    Rationale: Robust to class imbalance; emphasizes IoU rather than
    per-pixel accuracy. Soft Dice naturally extends to soft targets.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft Dice loss, averaged per (batch, channel) over the
        valid spatial region, then meaned over batch and channel."""
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)

        intersection = _spatial_sums(pred * target, mask)
        pred_sum = _spatial_sums(pred, mask)
        target_sum = _spatial_sums(target, mask)

        dice = (2.0 * intersection + _EPS) / (pred_sum + target_sum + _EPS)
        return 1.0 - dice.mean()


class FocalLoss(MaskedLoss):
    """Focal loss for handling hard negatives / imbalance.

    FL(pt) = -alpha * (1 - pt)^gamma * log(pt).
    Common: alpha=0.25, gamma=2.

    Rationale: Downweights easy negatives (high confidence, correct).
    Focuses on hard examples (low confidence or incorrect).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.

        Parameters:
            alpha: weighting factor (default 0.25).
            gamma: focusing exponent (default 2.0).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Generalizes the binary focal loss to soft targets by linearly
        blending the positive-class and negative-class terms with
        `target`/`(1 - target)` instead of a hard 0/1 selector.
        """
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)

        pred = pred.clamp(min=_EPS, max=1.0 - _EPS)
        pos_term = -self.alpha * target * (1.0 - pred) ** self.gamma * torch.log(pred)
        neg_term = (
            -(1.0 - self.alpha)
            * (1.0 - target)
            * pred**self.gamma
            * torch.log(1.0 - pred)
        )
        return _masked_mean(pos_term + neg_term, mask)


class TverskyLoss(MaskedLoss):
    """Tversky loss: generalization of Dice with FP/FN weighting.

    Tversky = TP / (TP + alpha*FP + beta*FN).
    Default: alpha=beta=0.5 → Dice. Increase alpha → penalize false positives more.

    Rationale: Useful for imbalanced detection (tune alpha/beta to balance
    precision vs recall needs).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """Initialize Tversky loss.

        Parameters:
            alpha: FP weight (default 0.5, set > 0.5 to penalize FP).
            beta: FN weight (default 0.5, set < 0.5 to allow FN).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Tversky loss, averaged per (batch, channel)."""
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)

        tp = _spatial_sums(pred * target, mask)
        fp = _spatial_sums(pred * (1.0 - target), mask)
        fn = _spatial_sums((1.0 - pred) * target, mask)

        tversky = (tp + _EPS) / (tp + self.alpha * fp + self.beta * fn + _EPS)
        return 1.0 - tversky.mean()


class BCEWithLogitsLoss(MaskedLoss):
    """Binary cross-entropy with logits (sigmoid + BCE, numerically stable).

    Rationale: Standard for binary classification; numerically stable.
    Supports soft targets (label smoothing, mixup).
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BCE loss.

        Note: pred is pre-sigmoid (logits); target is soft float [0, 1].
        """
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)
        elementwise = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return _masked_mean(elementwise, mask)


class MSELoss(MaskedLoss):
    """Mean squared error (regression).

    Rationale: Standard for regression (density mode). Penalizes outliers heavily.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)
        return _masked_mean((pred - target) ** 2, mask)


class WeightedMSELoss(MaskedLoss):
    """MSE with per-pixel weight proportional to target value.

    Loss = sum(weight * (pred - target)^2) over valid pixels.
    weight = target (or normalized target).

    Rationale: For density/count labels, emphasizes high-density regions
    where stream presence is clearest. Balances hard (dense) and easy
    (sparse) regions.
    """

    def __init__(self, normalize_weight: bool = True):
        """Initialize weighted MSE.

        Parameters:
            normalize_weight: if True, rescale weights (over valid pixels)
                to sum to 1, so the loss is a true weighted average
                independent of image size/pixel count. If False, weights
                are used as-is (`target`, clamped at 0) and the loss scales
                with both target magnitude and the number of valid pixels.
        """
        super().__init__()
        self.normalize_weight = normalize_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted MSE loss."""
        _check_shapes(pred, target)
        mask = _broadcast_mask(pred, valid_mask)

        # Weights come from the target itself (e.g. a stream star count),
        # which is never negative; clamp defensively in case a caller
        # passes a signed quantity anyway.
        weight = target.clamp(min=0.0) * mask.to(target.dtype)
        if self.normalize_weight:
            weight = weight / weight.sum().clamp(min=_EPS)

        return (weight * (pred - target) ** 2).sum()


_LOSS_REGISTRY: dict[str, type[MaskedLoss]] = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "tversky": TverskyLoss,
    "bce": BCEWithLogitsLoss,
    "mse": MSELoss,
    "weighted_mse": WeightedMSELoss,
}


def get_loss(name: str, **kwargs) -> MaskedLoss:
    """Factory function to instantiate loss by name.

    Parameters:
        name: Loss name ("dice", "focal", "tversky", "bce", "mse", "weighted_mse").
        **kwargs: Loss-specific keyword arguments.

    Returns:
        Instantiated MaskedLoss subclass.

    Raises:
        ValueError if name not recognized.

    Rationale: Central registry of available losses; makes config-based
    loss selection straightforward. "mse"/"weighted_mse" are the primary
    pair for this project's current default label_policy="stream_count"
    (see module docstring); the rest remain available for the
    binary/density label_policy options.
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name!r}. Available: {sorted(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name](**kwargs)
