"""U-Net for per-channel stream segmentation.

Architecture: encoder–decoder with skip connections.
Input: (B, in_channels, H, W) where in_channels = len(distance_moduli).
Output: (B, out_channels, H, W) per per-channel segmentation.

Head types (tied to label policy):
- sigmoid: binary classification (crossentropy, focal, dice).
- identity: regression (MSE, Poisson-style).
- softplus: smooth non-negative regression.

Rationale: Plain PyTorch, no hyrax imports. Reusable encoder for Phase 2
(embeddings + vector search). Keep it simple and explicit.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UNet(nn.Module):
    """U-Net for segmentation.
    
    Config:
        in_channels: number of input channels (typically len(distance_moduli)).
        out_channels: number of output channels (typically in_channels or 1).
        base_width: initial feature map width (default 32).
        depth: number of encoder/decoder blocks (default 4).
        head: output head activation ("sigmoid", "identity", "softplus").
    
    Rationale: Multi-scale processing via encoder–decoder captures both
    fine structure (thin streams) and context (local background).
    Skip connections preserve spatial detail.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_width: int = 32,
        depth: int = 4,
        head: str = "sigmoid"
    ):
        """Initialize U-Net.
        
        Parameters:
            in_channels: input channels.
            out_channels: output channels.
            base_width: feature map width at first level.
            depth: encoder/decoder depth.
            head: output activation type.
        
        Raises:
            ValueError if head is not recognized.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_width = base_width
        self.depth = depth
        self.head_type = head
        
        if head not in ("sigmoid", "identity", "softplus"):
            raise ValueError(f"Unknown head: {head}")
        
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters:
            x: (B, in_channels, H, W) input tensor.
        
        Returns:
            (B, out_channels, H, W) output tensor (head activation applied).
        
        Note: Invalid (masked) input pixels should be filled with a neutral
        value (e.g., 0 post-normalization) before forward pass. The network
        never sees NaNs; valid_mask only re-enters at the loss.
        """
        raise NotImplementedError
    
    def encoder(self, x: torch.Tensor):
        """Extract encoder features (for Phase 2 embeddings).
        
        Parameters:
            x: (B, in_channels, H, W) input tensor.
        
        Returns:
            Encoder bottleneck features (typically (B, base_width * 2^depth, H/2^depth, W/2^depth)).
        
        Rationale: Enables reuse of trained encoder for downstream tasks
        (similarity search after Phase 1 milestone).
        """
        raise NotImplementedError
