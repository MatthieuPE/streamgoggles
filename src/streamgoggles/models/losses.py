"""Segmentation losses accepting soft targets and optional valid_mask.

All losses accept:
- pred: model output (B, C, H, W).
- target: target labels (B, C, H, W), float or soft.
- valid_mask: optional (B, H, W) or (H, W) bool mask.

Masked reduction: compute loss only over valid_mask==True pixels.
This is NOT a class weight (we don't rebalance valid vs invalid);
it's a true EXCLUSION: invalid pixels contribute zero, not a weighted zero.

Rationale: Invalid pixels (outside survey footprint) should not affect training.
All losses must handle soft float targets [0, 1] for density labels.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        valid_mask: torch.Tensor | None = None
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
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute soft Dice loss.
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


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
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


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
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute Tversky loss.
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


class BCEWithLogitsLoss(MaskedLoss):
    """Binary cross-entropy with logits (sigmoid + BCE, numerically stable).
    
    Rationale: Standard for binary classification; numerically stable.
    Supports soft targets (label smoothing, mixup).
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute BCE loss.
        
        Note: pred is pre-sigmoid (logits); target is soft float [0, 1].
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


class MSELoss(MaskedLoss):
    """Mean squared error (regression).
    
    Rationale: Standard for regression (density mode). Penalizes outliers heavily.
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute MSE loss.
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


class WeightedMSELoss(MaskedLoss):
    """MSE with per-pixel weight proportional to target value.
    
    Loss = sum(weight * (pred - target)^2) over valid pixels.
    weight = target (or normalized target).
    
    Rationale: For density labels, emphasizes high-density regions
    where stream presence is clearest. Balances hard (dense) and easy
    (sparse) regions.
    """
    
    def __init__(self, normalize_weight: bool = True):
        """Initialize weighted MSE.
        
        Parameters:
            normalize_weight: if True, scale weights to sum to 1.
        """
        super().__init__()
        self.normalize_weight = normalize_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute weighted MSE loss.
        
        Raises:
            NotImplementedError (stub).
        """
        raise NotImplementedError


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
    loss selection straightforward.
    """
    raise NotImplementedError
