"""Minimal training loop (plain PyTorch, no hyrax).

Used by the notebook and standalone scripts. Handles:
- epoch loop (forward, backward, optimize)
- checkpointing via ModelStore
- metric logging
- optional automatic mixed precision (AMP)
- simple learning rate scheduling (optional)

Rationale: Keeps training code simple and debuggable. No framework
magic; every step is explicit and understandable.
"""

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PlainTrainer:
    """Minimal training loop for U-Net.
    
    Attributes:
        model: torch.nn.Module (e.g., UNet).
        optimizer: torch.optim.Optimizer.
        loss_fn: Loss function (e.g., DiceLoss).
        device: "cpu" or "cuda".
        amp: if True, use torch.cuda.amp for mixed precision.
    
    Rationale: Explicit, single-threaded loop. No async loading, no parallel
    GPUs. Easy to understand, debug, and profile. For the first milestone.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        amp: bool = False
    ):
        """Initialize trainer.
        
        Parameters:
            model: torch.nn.Module instance.
            optimizer: torch.optim.Optimizer.
            loss_fn: Loss function (should accept valid_mask).
            device: "cpu" or "cuda".
            amp: if True, use torch.cuda.amp.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.amp = amp
        self.scaler = None
        if amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, dl: DataLoader) -> dict:
        """Run one epoch on training data.
        
        Parameters:
            dl: DataLoader yielding (map_stack, label_stack, valid_mask, metadata).
        
        Returns:
            dict with keys:
                loss: average loss (float)
                n_batches: number of batches processed
                lr: current learning rate
        
        Rationale: Minimal logging; relies on caller for periodic validation/checkpointing.
        """
        raise NotImplementedError
    
    def validate(self, dl: DataLoader, metrics_fn: Callable | None = None) -> dict:
        """Run one validation pass.
        
        Parameters:
            dl: DataLoader (eval set).
            metrics_fn: optional callable(pred, target, valid_mask) -> dict of metrics.
        
        Returns:
            dict with keys:
                loss: average loss
                *metrics: from metrics_fn if provided
        
        Rationale: No backprop; just inference and metric computation.
        """
        raise NotImplementedError
    
    def train(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader | None = None,
        epochs: int = 10,
        model_store: "ModelStore" = None,
        config: dict | None = None,
        save_every: int = 1,
        callbacks: list = None
    ) -> dict:
        """Full training loop: epochs, validation, checkpointing, logging.
        
        Parameters:
            train_dl: Training DataLoader.
            val_dl: Validation DataLoader (optional; if None, no validation).
            epochs: Number of epochs.
            model_store: ModelStore instance (optional; if provided, save checkpoints).
            config: Configuration dict (saved with checkpoint).
            save_every: Save checkpoint every N epochs.
            callbacks: Optional list of callable(epoch, trainer, metrics) hooks.
        
        Returns:
            dict with keys:
                train_losses: list of per-epoch train loss
                val_losses: list of per-epoch val loss (or None if no validation)
                final_checkpoint_path: path to last saved checkpoint
                epochs_run: number of epochs trained
        
        Raises:
            ValueError if config missing required keys.
        """
        raise NotImplementedError
