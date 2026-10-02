# Networks that answer with lines

The U-Net of {doc}`datasets_and_models` answers **pixel by pixel**: for every
pixel of a window, the probability that it belongs to a stream. This page
describes a second design, which answers **line by line**: for every
straight line through the window, the probability that a stream lies along
it. The line is computed with a *Hough transform* built into the network.
This page explains the method, how each network is plugged, what goes in and
what comes out, and follows real examples through it. How well it works on
the real DES sky is in {doc}`../experiments/line_model/window_level`.

```{note}
Code: {py:mod}`streamgoggles.models.hough` (the transform, the line model,
its training target), {py:func}`streamgoggles.models.build_model` (choosing a
model by options), {py:class}`~streamgoggles.datasets.transforms.ResidualNormalizer`
(the S/N inputs). The figures of this page are made by
`scripts/experiments/line_model/run.py figures`.
```

## Why lines

A stream is long and thin. Its significance comes from **adding up** many
faint pixels along its track: a stream that adds a few stars per pixel to a
background of twenty is invisible pixel by pixel, but its excess summed
over a hundred pixels along the track, against the noise of the background
summed over the same pixels, can stand at many standard deviations. That
sum is what a matched-filter search does along a known track.

A per-pixel network cannot do that sum: each output pixel only sees a small
neighbourhood (for the U-Net used here, 90% of what decides a pixel lies
within 1.5° of it, see {doc}`../experiments/real_des/labels_normalization`).
It acts as a *local* detector — it finds a stream where a few degrees of it
stand out — and on the real sky it needs about twice the input S/N the
matched filter needs along the track. Answering with lines moves the sum
inside the network.

## The Hough transform

### Lines in a window

Every straight line in a window can be written in its *normal form*

```{math}
x \cos\theta + y \sin\theta = \rho ,
```

with `(x, y)` the pixel's position from the window's centre (`x` along
columns, `y` along rows), `θ` the direction of the line's normal, between 0°
and 180°, and `ρ` the line's distance from the centre, positive or negative.
Each pair `(θ, ρ)` is one line. The grid used here has **90 angles** (2°
apart) and **69 offsets** (2 pixels, 0.23° apart, covering the window's
diagonal), so a window of 96 × 96 pixels has 90 × 69 = 6,210 lines. Lines
with fewer than 20 pixels inside the window (across its corners) are left
out — they are drawn white in the figures below.

The windows are **gnomonic** projections of the sky, in which every great
circle is a straight line: a stream following a great circle is exactly one
line of the grid (a stream with a curved track, approximately).

`θ` and `θ + 180°` are the same direction, so the line `(θ, ρ)` near 180°
reappears near 0° as `(θ − 180°, −ρ)`: a near-vertical stream shows up at
both ends of the `θ` axis.

### Line sums

The transform turns an image `I` into one number per line, the sum of the
image along it, divided by the square root of the line's number of pixels:

```{math}
H(\theta, \rho) = \frac{1}{\sqrt{n(\theta, \rho)}} \sum_{p \,\in\, \text{line}(\theta, \rho)} I(p) .
```

Each pixel votes, for each angle, into the two offsets nearest its own
`x cos θ + y sin θ` (with linear weights), so `n(θ, ρ)` is the line's total
weight. Dividing by `√n` puts every line on the same footing: on white noise
of unit variance, every line sum has about unit variance, whatever its
length. And on an **S/N map** — each pixel's excess over the background,
over the background's square root — `H(θ, ρ)` *is* the line's matched-filter
S/N: the excess summed along the line over the noise summed along it.

The transform is a fixed linear map, a sparse matrix of 6,210 lines × 9,216
pixels: {py:class}`~streamgoggles.models.hough.HoughLines` applies it to a
numpy image, {py:class}`~streamgoggles.models.hough.HoughTransform` to
torch tensors, channel by channel, so gradients flow through it during
training.

### What shapes look like over (θ, ρ)

```{image} figures/line_model/hough_toy.png
:alt: Four toy windows and their line sums: a line gives one peak, a point a sinusoid, a short segment a peak spread in theta, a faint line in noise a peak at 6.4 sigma
:width: 100%
```

*Top: four windows. Bottom: their line sums over (θ, ρ); orange circle: the
true line.*

- **A line** gives one peak, at its own `(θ, ρ)`.
- **A point** (or a compact blob) lies on every line through it, and those
  lines form the sinusoid `ρ = x₀ cos θ + y₀ sin θ`. A blob bright enough
  makes all of them bright — which is why a plain line search fires on
  globular clusters and dwarf galaxies.
- **A short segment** gives a peak spread in `θ`: many directions pass
  through most of it.
- **A faint line in noise** (bottom right), 2 pixels wide and 0.45σ per
  pixel — invisible by eye — stands at 6.4σ over the noise of the line sums.
  That is the integration the per-pixel network lacks.

## How each network is plugged

```{image} figures/line_model/architecture.png
:alt: Four pipelines from a window to an answer: the per-pixel U-Net, the line model, the per-pixel U-Net with a line search, and the matched-filter line search, with the shapes in between
:width: 100%
```

All four start from the same window, the 96 × 96 pixel window of
{doc}`data_generation`, at one queried distance:

| | input | output | trained on | in the experiment |
|---|---|---|---|---|
| **per-pixel U-Net** | 7 × 96 × 96 | 1 × 96 × 96: probability per pixel | count or band label, per pixel | `count/window` |
| **line model** | 7 × 96 × 96 | 1 × 90 × 69: probability per line | the band label, turned into lines | `hough/band`, `hough/band residual` |
| **per-pixel U-Net + line search** | 7 × 96 × 96 | 90 × 69: score per line | — (the per-pixel models, unchanged) | `count/window x4 lines` |
| **matched-filter line search** | counts at the queried distance | 90 × 69: S/N per line | — (no network) | scored alongside every line configuration |

**The input** (`QueryDistanceTransform`): for a queried distance modulus
`q`, seven channels — the isochrone matched filter's maps at `q − 0.5`, `q`
and `q + 0.5` (so a small distance gradient can be followed), the decoy box
(a colour-magnitude box off the isochrone: what the background looks like),
and three constant maps holding the three distances, so the same network
serves every distance.

**The per-pixel U-Net** ({py:class}`~streamgoggles.models.unet.UNet`, depth
2, 12 base channels, sigmoid head) maps those 7 channels to one
probability per pixel; the label, the losses and how it is trained are in
{doc}`datasets_and_models`.

**The line model**
({py:class}`~streamgoggles.models.hough.HoughUNet`), step by step:

| step | what it does | shape |
|---|---|---|
| input | the 7 channels | 7 × 96 × 96 |
| backbone | the same U-Net, with 8 output channels and no activation: learned local features, with **no per-pixel label and no per-pixel decision** — nothing is squashed or thresholded before the sums | 8 × 96 × 96 |
| + the input | the 7 input channels are appended untouched, so their own line sums always reach the end | 15 × 96 × 96 |
| Hough transform | every channel summed along every line, over √n (fixed, not trained) | 15 × 90 × 69 |
| over the lines | two convolutions over (θ, ρ), 3 angles × 5 offsets each (the second sees 2° of offsets, so a wide stream's signal, spread over neighbouring lines, is gathered back), batch normalization and ReLU | 16 × 90 × 69 |
| head | a 1 × 1 convolution: one logit per line; the sigmoid makes it a probability | 1 × 90 × 69 |

Because the input reaches the lines untouched, the line model always has
the matched filter's line sums of its own input; the backbone can only add
to them. With the window-normalized inputs of the per-pixel models, those
sums are not S/N; with S/N inputs (below) they are exactly the matched
filter's line search.

**The per-pixel U-Net with a line search** is a combination, not a new
network: the per-pixel models' output, as a logit minus its median over the
window, summed along every line over √n. It tests whether adding a line
search *after* the per-pixel network is enough.

**The matched-filter line search** has no network at all: the counts at the
queried distance minus a smooth local background, over the background's
square root, summed along every line. It is the classical baseline every
network has to beat.

## The inputs: why S/N maps

What reaches the lines has to be an S/N for its sums to mean anything. Two
normalizations of each channel, computed from the window alone:

- `WindowNormalizer` (the per-pixel models'): the channel minus its mean,
  over its spread, in the window. A window's mean and spread are not its
  background and noise: a density gradient across the window survives, and
  the line sums of a window with a gradient are dominated by it.
- `ResidualNormalizer` ("S/N inputs"): the channel's excess over a **local
  background plane**, over the plane's square root,

  ```{math}
  x'(p) = \frac{x(p) - b(p)}{\sqrt{b(p)}} ,
  ```

  with `b(p)` a plane fitted to the counts around `p` over valid pixels
  alone, weighted by a Gaussian of 1.8°. A plane rather than a weighted mean
  follows gradients up to the window's edges and the masks' edges, where a
  mean would leave a band of residual along every edge — a straight feature
  a line search would find. The plane is fitted on the window as observed,
  stream included, so a wide stream gives part of its excess to the
  background: a narrow one keeps about 90% of its peak, one of Gaussian
  width 1.1° about half.

## Training a line model: from the band to its lines

The label is the band label of {doc}`../experiments/real_des/labels_normalization`
(the stream's band, one width either side of its track, where it stands at
S/N ≥ 2 at the queried distance), turned into lines by
{py:func}`~streamgoggles.models.hough.hough_target`: count the band's pixels
on every line, keep the lines holding at least 80% of the best line's
count, and add their neighbours one step away in `θ` and `ρ` (a line one
step off is still the stream). A window whose stream is not visible at the
queried distance, or that has no stream, has no positive line.
{py:class}`~streamgoggles.models.hough.HoughTargetTransform` does it on the
fly, after the query view: the label `1 × 96 × 96` becomes the target
`1 × 90 × 69`, and the valid mask becomes the valid lines.

The loss is the cross-entropy over the valid lines, with the head's bias
started at the share of positive lines; 30% of the training windows have no
stream, since a window may now answer "no line".

## Reading the answer

A line model, or any of the line searches, gives one score per line of each
window. Two ways to read it:

- **Without the track** (a search): report every line scoring above a
  level, set on stream-free windows of the calibration sky so that 1% of
  them have a line above it. A line is a track: `(θ, ρ)` in the window,
  and through the window's gnomonic projection, a great circle on the sky.
- **Along a known track**: the best score among the track's own lines,
  against the same lines in stream-free windows.

The stream-free windows must lie on the calibration sky: a line search
integrates over a whole window, so a single straight feature of the sky —
the edge of a mask with real stars beyond it, as along Sagittarius — sets
its false-alarm level (see {doc}`../experiments/line_model/window_level`).

## Worked examples

Three DES 2018 streams, each injected with its Table 1 parameters at its
first place of the evaluation on fold 1 of the real sky, in the search
window holding the longest stretch of it, followed through both networks
(the four per-pixel models, and the line model on S/N inputs). Top row:
the counts, the S/N input, the per-pixel network's answer, and the line
network's best line back on the sky. Bottom row: the S/N input's own line
sums, the lines along the copy's track (orange contour), the line network's
probability per line (white cross: its best line), and the reading.

**Jhelum**, the widest stream (1.16°), near (m−M 15.6): invisible in the
counts; the per-pixel network sees nothing (0.01 at most); the line network's
best line is the stream's (0.67, above the 0.43 that 1% of stream-free
windows reach).

```{image} figures/line_model/example_jhelum.png
:alt: Jhelum followed through both networks: found by the line network, not by the per-pixel network
:width: 100%
```

**Wambelong**, the faintest (input S/N 8.6): the per-pixel network catches
fragments of it (0.65 at most); the line network finds its line (0.58,
above 0.47).

```{image} figures/line_model/example_wambelong.png
:alt: Wambelong followed through both networks: found by the line network, in fragments by the per-pixel network
:width: 100%
```

**Tucana III**, the shortest (4.8°) and among the brightest: plain to see;
the per-pixel network flags it fully, and the S/N input's own line sums
peak on its lines — but the line network gives them 0.05. This is the line
model's known failure on the real sky: it misses the short bright streams,
perhaps confusing them with the compact overdensities of the stream-free
sky it learned to ignore.

```{image} figures/line_model/example_tucana_iii.png
:alt: Tucana III followed through both networks: missed by the line network, found by the per-pixel network and by the line sums of its input
:width: 100%
```

## Choosing a model by options

The per-pixel U-Net is unchanged and stays the default; a configuration
chooses its model by options alone.

```python
from streamgoggles.models import build_model

pixels = build_model(7, depth=2, base_width=12, head="sigmoid")      # kind="unet"
lines = build_model(7, 96, kind="hough", depth=2, base_width=12)     # one answer per line

x = torch.randn(4, 7, 96, 96)
pixels(x).shape   # (4, 1, 96, 96): probability per pixel
lines(x).shape    # (4, 1, 90, 69): logit per line
```

The transform alone, on any S/N image:

```python
from streamgoggles.models.hough import HoughLines, hough_target

grid = HoughLines(96, 96)             # 90 angles x 69 offsets, lines >= 20 pixels
sums = grid(snr_image)                # (90, 69): each line's matched-filter S/N
target = hough_target(band_mask, grid)   # the band's lines, as a 0/1 map
```

In the line-model experiment (`scripts/experiments/line_model/run.py`), a
configuration is a dictionary; `hough` makes it a line model, `lines` a
per-pixel ensemble searched along lines, and `evaluate` scores both per
window (the per-pixel models themselves are trained and scored per pixel by
`scripts/experiments/real_des/patches.py`):

```python
"hough/band residual": {
    "label": "band",                  # the band label, turned into lines
    "normalizer": "residual",         # S/N inputs
    "loss": "bce",
    "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
    "training": {"background_fraction": 0.3},
},
"count/window x4 lines": {            # four per-pixel models + a line search
    "label": None, "normalizer": "window",
    "parts": ["count/window"], "seeds": [42, 43, 44, 45], "lines": True,
},
```

## Limits

- **Straight lines in 11° windows.** A stream longer than the window is
  seen in pieces, each window scoring its own stretch; a curved track is
  approximated by its best line.
- **The angle is not periodic** in the convolutions over (θ, ρ): a line near
  0° and the same line near 180° are not neighbours there.
- **A line search sees every straight feature**: the edges of masks with
  stars beyond them, and every line through a bright blob. Its false-alarm
  level has to be set on sky free of both.
- **On the real sky**, the line model on S/N inputs finds near, faint
  streams better than anything before it, but misses the short bright ones
  ({doc}`../experiments/line_model/window_level`).
