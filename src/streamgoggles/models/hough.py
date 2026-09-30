"""A U-Net whose answer is a line: the Hough transform built into the network.

The per-pixel U-Net acts as a local detector: it flags a stream where a few
degrees of it stand out, and misses faint streams that only stand out when
summed along their whole length -- which is what a matched filter along a
known track does. Here the integration is part of the network:

1. a U-Net backbone maps the window to a few local feature maps. It has no
   per-pixel label and no per-pixel decision: nothing is thresholded or
   squashed before the sums, and it is trained only through the line loss,
   so a stream no single pixel shows is not lost at this step;
2. a fixed Hough transform sums each feature map -- and the input channels
   themselves, passed around the backbone -- along every straight line
   through the window, divided by the square root of the line's number of
   pixels (for white noise, about a line's S/N). The matched filter's own
   line sums are thus always there; the backbone can only add to them;
3. a small convolutional network in Hough space turns these line sums into
   one map of logits: "a stream lies along the line (theta, rho)". Its
   kernels span 5 rho bins (10 pixels, 1.1 degrees), two layers 2 degrees,
   so a wide stream's signal spread over neighbouring lines is summed back.

Windows are gnomonic projections, in which great circles are straight lines,
so a stream along a great circle is exactly one line of the transform.

Lines: ``x cos(theta) + y sin(theta) = rho`` in pixels from the window's
centre (x along columns, y along rows), theta in [0, pi) with ``n_theta``
steps, rho in ``rho_step`` pixel bins covering the window's diagonal. Each
pixel votes, for each theta, to its two nearest rho bins (linear weights).

The label: the Hough transform of the window's label mask at the queried
distance; the lines holding at least ``fraction`` of its best line's labelled
pixels, and their neighbours, are the positives; a window with an empty label
has none
(`hough_target`, `HoughTargetTransform`).
"""

import numpy as np
import torch
from scipy import ndimage, sparse
from torch import nn

from streamgoggles.models.unet import UNet


def hough_matrix(height, width, n_theta=90, rho_step=2.0):
    """Sparse (n_theta * n_rho, height * width) voting matrix, and its grids.

    Returns:
        (matrix, thetas, rhos): a scipy CSR matrix whose row ``t * n_rho + r``
        holds each pixel's weight on the line (thetas[t], rhos[r]); the theta
        grid (radians) and the rho bin centres (pixels).
    """
    y, x = np.mgrid[0:height, 0:width]
    x = (x - (width - 1) / 2).ravel()
    y = (y - (height - 1) / 2).ravel()
    half = np.hypot(height, width) / 2
    rhos = np.arange(-half, half + rho_step, rho_step)
    thetas = np.arange(n_theta) * np.pi / n_theta
    rows, cols, weights = [], [], []
    pixels = np.arange(x.size)
    for t, theta in enumerate(thetas):
        position = (x * np.cos(theta) + y * np.sin(theta) - rhos[0]) / rho_step
        low = np.floor(position).astype(int)
        upper = position - low
        for offset, weight in ((0, 1 - upper), (1, upper)):
            rows.append(t * rhos.size + low + offset)
            cols.append(pixels)
            weights.append(weight)
    matrix = sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_theta * rhos.size, height * width),
    )
    return matrix, thetas, rhos


class HoughLines:
    """The line grid and its (numpy) transform: line sums of an (H, W) image
    over sqrt(line length). Plain numpy, so data-loading workers carry it.

    Attributes:
        thetas, rhos: the grids (radians, pixels).
        shape: (n_theta, n_rho).
        lengths: pixels on each line (sum of its weights).
        valid: lines with at least ``min_pixels`` pixels in the window; the
            shorter ones are left out of the loss and of the scoring.
        matrix: scipy CSR, rows divided by sqrt(length): white noise gives
            line sums of about unit variance.
    """

    def __init__(self, height, width, n_theta=90, rho_step=2.0, min_pixels=20):
        matrix, self.thetas, self.rhos = hough_matrix(height, width, n_theta, rho_step)
        self.shape = (n_theta, self.rhos.size)
        self.lengths = np.asarray(matrix.sum(axis=1)).reshape(self.shape)
        self.valid = self.lengths >= min_pixels
        norm = 1 / np.sqrt(np.maximum(self.lengths.ravel(), 1.0))
        self.matrix = (sparse.diags(norm) @ matrix).tocsr()

    def __call__(self, image):
        """Line sums of one (H, W) image, (n_theta, n_rho)."""
        return (self.matrix @ np.asarray(image, float).ravel()).reshape(self.shape)


class HoughTransform(nn.Module):
    """`HoughLines` on tensors, per channel: (B, C, H, W) -> (B, C, n_theta,
    n_rho)."""

    def __init__(self, height, width, n_theta=90, rho_step=2.0, min_pixels=20):
        super().__init__()
        self.grid = HoughLines(height, width, n_theta, rho_step, min_pixels)
        self.shape = self.grid.shape
        coo = self.grid.matrix.tocoo()
        self.register_buffer(
            "matrix",
            torch.sparse_coo_tensor(
                np.vstack([coo.row, coo.col]),
                coo.data.astype(np.float32),
                coo.shape,
            ).coalesce(),
            persistent=False,
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        flat = x.reshape(batch * channels, height * width).T
        lines = torch.sparse.mm(self.matrix, flat)
        return lines.T.reshape(batch, channels, *self.shape)


def hough_target(label, grid, fraction=0.8):
    """Positive lines for a (H, W) label mask (`grid`: a `HoughLines`): those
    holding at least ``fraction`` of the best line's labelled pixels, and
    their neighbours one step away in theta and rho. All zeros if the label
    is empty."""
    mask = np.asarray(label) > 0.5
    if not mask.any():
        return np.zeros(grid.shape, np.float32)
    # line sums are over sqrt(length); undo it to count labelled pixels
    counts = grid(mask) * np.sqrt(np.maximum(grid.lengths, 1.0))
    counts = np.where(grid.valid, counts, 0.0)
    best = counts >= fraction * counts.max()
    # one cell either way in theta and rho: a line a step off is still the stream
    best = ndimage.binary_dilation(best, structure=np.ones((3, 3), bool))
    return (best & grid.valid).astype(np.float32)


class HoughTargetTransform:
    """Wraps a view (e.g. `QueryDistanceTransform`): its one-channel label
    becomes the Hough target, its valid mask the valid lines (`grid`: a
    `HoughLines`)."""

    def __init__(self, inner, grid, fraction=0.8):
        self.inner = inner
        self.grid = grid
        self.fraction = fraction

    def __call__(self, sample):
        out = self.inner(sample)
        out["label_stack"] = hough_target(
            out["label_stack"][0], self.grid, self.fraction
        )[None]
        out["valid_mask"] = self.grid.valid.copy()
        return out


class HoughUNet(nn.Module):
    """U-Net features and the input -> line sums -> logits per line
    (B, 1, n_theta, n_rho).

    ``head_conv`` is the last layer, as in `UNet`, so the same prior-bias
    initialization applies.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        features=8,
        depth=2,
        base_width=12,
        n_theta=90,
        rho_step=2.0,
        min_pixels=20,
        hough_width=16,
        **_,
    ):
        super().__init__()
        self.backbone = UNet(
            in_channels=in_channels,
            out_channels=features,
            depth=depth,
            base_width=base_width,
            head="identity",
        )
        self.hough = HoughTransform(
            image_size, image_size, n_theta, rho_step, min_pixels
        )
        self.lines = nn.Sequential(
            nn.Conv2d(
                features + in_channels, hough_width, kernel_size=(3, 5), padding=(1, 2)
            ),
            nn.BatchNorm2d(hough_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(hough_width, hough_width, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(hough_width),
            nn.ReLU(inplace=True),
        )
        self.head_conv = nn.Conv2d(hough_width, 1, kernel_size=1)

    def forward(self, x):
        features = torch.cat([self.backbone(x), x], dim=1)
        return self.head_conv(self.lines(self.hough(features)))
