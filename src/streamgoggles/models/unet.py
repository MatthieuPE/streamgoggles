"""U-Net for per-channel stream segmentation.

Architecture: encoder-decoder with skip connections.
Input: (B, in_channels, H, W) where in_channels = n_distances * n_filters
(one channel per (filter, distance_modulus) pair -- see
Sample.metadata["channels"] for the exact, explicit ordering; the pivot
notes in PLAN.md section 6.3 describe why this is no longer just
len(distance_moduli)).
Output: (B, out_channels, H, W), per-channel by default (out_channels ==
in_channels), matching label_stack's own shape under label_policy=
"stream_count" (Sample.__post_init__ requires map_stack.shape ==
label_stack.shape).

Head types (tied to label policy):
- sigmoid: binary classification (crossentropy, focal, dice).
- identity: regression (MSE, Poisson-style).
- softplus: smooth non-negative regression -- the natural fit for the
  current default label_policy="stream_count" (a non-negative count), but
  not forced as the default here so the signature stays exactly what
  PLAN.md section 3.13 specifies; callers targeting stream_count should
  pass head="softplus" explicitly.

Rationale: Plain PyTorch, no hyrax imports. Reusable encoder for Phase 2
(embeddings + vector search). Keep it simple and explicit.

Small/odd input sizes: this project's PixelizationSpec.image_size_pix is
often much smaller than typical U-Net inputs (e.g. 15x15, 20x20) and not
always a power of 2. Downsampling uses max-pool with ceil_mode=True (so an
odd spatial size rounds up instead of vanishing), and upsampling resizes
back to the matching skip connection's exact spatial size via
F.interpolate(..., size=skip.shape[-2:]) rather than a fixed scale factor,
so skip concatenation never needs cropping/padding and any depth is safe
down to a 1x1 bottleneck.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers, same spatial size in/out."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net for segmentation.

    Config:
        in_channels: number of input channels (n_distances * n_filters).
        out_channels: number of output channels (typically in_channels or 1).
        base_width: initial feature map width (default 32).
        depth: number of encoder/decoder blocks (default 4).
        head: output head activation ("sigmoid", "identity", "softplus").

    Rationale: Multi-scale processing via encoder-decoder captures both
    fine structure (thin streams) and context (local background).
    Skip connections preserve spatial detail.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_width: int = 32,
        depth: int = 4,
        head: str = "sigmoid",
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

        # enc_widths[i] = feature width at encoder stage i (before pooling).
        enc_widths = [base_width * (2**i) for i in range(depth)]
        bottleneck_width = base_width * (2**depth)

        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.encoder_blocks = nn.ModuleList()
        prev_width = in_channels
        for width in enc_widths:
            self.encoder_blocks.append(ConvBlock(prev_width, width))
            prev_width = width

        self.bottleneck = ConvBlock(prev_width, bottleneck_width)

        # Decoder stage i consumes the upsampled previous-stage output
        # (decoder_widths[i]) concatenated with skip i (enc_widths[i]).
        decoder_widths = [bottleneck_width, *enc_widths[:0:-1]]
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            in_width = decoder_widths[depth - 1 - i] + enc_widths[i]
            self.decoder_blocks.append(ConvBlock(in_width, enc_widths[i]))

        self.head_conv = nn.Conv2d(enc_widths[0], out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        h = x
        for block in self.encoder_blocks:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        return h, skips

    def _decode(self, h: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for block, skip in zip(self.decoder_blocks, reversed(skips), strict=True):
            h = F.interpolate(
                h, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        return h

    def _apply_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "sigmoid":
            return torch.sigmoid(x)
        if self.head_type == "softplus":
            return F.softplus(x)
        return x

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
        h, skips = self._encode(x)
        h = self._decode(h, skips)
        h = self.head_conv(h)
        return self._apply_head(h)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features (for Phase 2 embeddings).

        Parameters:
            x: (B, in_channels, H, W) input tensor.

        Returns:
            Encoder bottleneck features (typically (B, base_width * 2^depth, H/2^depth, W/2^depth) --
            with ceil-mode pooling on non-power-of-2 inputs, the exact
            spatial size is ceil-divided at each of the `depth` pooling
            stages rather than an exact power-of-2 division).

        Rationale: Enables reuse of trained encoder for downstream tasks
        (similarity search after Phase 1 milestone).
        """
        h, _ = self._encode(x)
        return h
