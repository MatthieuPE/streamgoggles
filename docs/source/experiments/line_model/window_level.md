# A network that answers with lines

{doc}`../real_des/labels_normalization` ended on one diagnosis: the per-pixel network
acts as a *local* detector. It flags a stream where a few degrees of it stand
out, and no change of label, loss, depth or training range made it integrate
a faint stream along its length — which is what the matched filter does,
summing counts along a known track. This page builds that integration into the
network: the network answers with **lines** through the window rather than
pixels, and is scored as a search would use it, without the track.

Branch `hough-window-model`; code in `src/streamgoggles/models/hough.py` and
`scripts/experiments/line_model/run.py` (sharing the skies of
`scripts/experiments/real_des/patches.py`); everything on the real DES Y6 sky,
trained on fold 0 and scored on fold 1, like {doc}`../real_des/labels_normalization`.

**In short.** Building the integration into the network works for near
streams, once its inputs are matched-filter S/N maps: without knowing the
track, the line network finds half of the near copies at an input S/N of
7.9 — against 12.1 for the per-pixel network's test (which does know the
track) and 12.5 for the matched filter's own line search — and it finds
Wambelong, which every other method misses. Along a known track it reaches
the matched filter's line sums (6.0 against 6.2). It does not yet for distant
streams (12.8 against 10.0), because it misses the two brightest short ones,
Tucana III and Willka Yaku. Fed the per-window normalized maps of the
per-pixel network instead, the same line network is worse than the matched
filter on every count: what reaches the lines has to be an S/N. And the
matched filter's line search alone, a classical Hough search, is a strong
blind baseline: 78% of the near and 99% of the distant DES 2018 copies, with
one false line per hundred stream-free windows.

## The model

How the line model works — the Hough transform, how each network is
plugged, their inputs and outputs, and worked examples — is explained in
the guide, {doc}`../../narrative/line_model`. In brief:

```text
window (7 channels) ─► U-Net backbone ─► 8 feature maps ─┐
        └──────────────────── the 7 input channels ───────┤
                                                          ▼
                          Hough transform: sum of each map along every line,
                          over √(line length)  →  15 maps over (θ, ρ)
                                                          ▼
                          2 convolutions over (θ, ρ)  →  logit per line
```

- **No per-pixel decision.** The backbone is the same small U-Net (depth 2,
  12 base channels), but it has no per-pixel label and no per-pixel output:
  its 8 feature maps are neither squashed nor thresholded, and it learns only
  through the line loss. A stream no single pixel shows is not lost before
  the sums.
- **The input reaches the lines untouched.** Next to the features, the input
  channels themselves are summed along the lines, so the network always has
  its input's own line sums — a matched-filter line search — and the
  backbone can only add to them.
- **Lines.** In a gnomonic window a great circle is a straight line, so a
  stream along one is exactly one line `x cos θ + y sin θ = ρ`: 90 angles
  (2°) and 69 offsets (2 pixels, 0.23°), lines with fewer than 20 pixels in
  the window left out. Each sum is divided by √(pixels on the line), so on
  white noise every line has the same spread. The two convolutions over
  (θ, ρ) span 5 offsets each (1.1°, then 2°), so a wide stream's signal,
  spread over neighbouring lines, is gathered back.
- **Label.** The band label (the stream's band where it stands at S/N ≥ 2 at
  the queried distance) turned into lines: those holding at least 80% of the
  best line's labelled pixels, and their neighbours one step away. A window
  with no visible stream has no positive line.
- **Training.** Cross-entropy over lines, 30% of windows stream-free (a
  window may now answer "no line"), the quick tier (4,800 windows, two
  seeds, about 17 minutes each).
- **Inputs: two normalizations.** `hough/band` takes the per-pixel models'
  inputs, each channel standardized by its own mean and spread in the window
  (`WindowNormalizer`). `hough/band residual` takes matched-filter S/N maps:
  each channel's excess over a local background plane, over that plane's
  square root (`ResidualNormalizer`; the plane is a Gaussian-weighted local
  linear fit, 1.8°, over valid pixels alone, so it follows density gradients
  up to the window's and the masks' edges, where a weighted mean would leave
  a band of residual along every edge — a straight feature). Then the input's
  own line sums, which reach the lines untouched, *are* the matched filter's
  line search, and the network starts from it. The plane is fitted on the
  window as observed, stream included: a narrow stream keeps about 90% of
  its peak, one of Gaussian width 1.1° (10 pixels) about half.

The model is chosen by options: `streamgoggles.models.build_model(kind=...)`
builds the per-pixel `UNet` (`"unet"`, the default, unchanged) or the
`HoughUNet` (`"hough"`); in the experiment, a configuration with a `hough`
entry is a line model, one with `lines` is a per-pixel ensemble searched
along lines (the combination below), the others per-pixel models as before.

## How it is scored

The same DES 2018 copies as in {doc}`../real_des/labels_normalization` — each stream at
its Table 1 parameters, 8 places on fold 1, at full brightness and 1 and 1.5
mag/arcsec² fainter — land on exactly the same places (checked copy by copy),
so the methods compare copy by copy. Each window covering a copy (the
half-overlapping tiles of a search), if it holds at least 4° of it, answers
with one score per line; the lines "along the track" are those the copy's
band makes there (the label's rule).

Three scorers, all on the same windows:

- the **line network** (the mean probability of its two seeds);
- the **matched filter's line search**: the counts at the queried distance
  minus a smooth local background, over its square root, summed along each
  line over √(length) — a classical Hough search, no network;
- the **per-pixel network with a line search on top** (the combination):
  the four count-label models' mean output, as a logit minus its window
  median, summed along each line the same way.

Two tests, against 600 stream-free windows per queried distance:

- **without the track (blind)**: a copy is found if one of its lines scores
  above what the best line of a stream-free window exceeds in 1% of them —
  a search reporting every line above that level, with one false line per
  hundred stream-free windows;
- **along the known track**: in the window holding the longest stretch of
  the copy, its best line against the same lines in the stream-free windows,
  found when at most 1/201 of them score as high (the level of the band
  tests of {doc}`../real_des/labels_normalization`).

The per-pixel network's current test is not blind — it counts flagged pixels
near the true track and compares them with the same band placed elsewhere —
so it is shown with the blind tests for reference only.

### What the stream-free sky holds

A line search integrates over a whole window, so a single straight feature
of the "stream-free" sky decides its false-alarm level. With stream-free
windows only *centred* on the calibration sky, the line network's 1% level
at m−M 17 was a probability of 0.94, and the matched filter's an S/N of 13.7
— while the median window's best line scores 0.10 and 3.3.

```{image} ../figures/line_model/hough_null_windows_fold0.png
:alt: Three stream-free windows with their strongest lines: two along the edge of the Sagittarius mask, one through a compact overdensity near a masked dwarf
:width: 100%
```

*The stream-free windows where the lines are strongest, at m−M 17. Top two:
the line network's — the best line (orange) runs along the edge of the
Sagittarius mask, where the isochrone channel holds a band of excess stars
the decoy does not: Sagittarius's wing, past its 6-degree mask, at about
its distance. Bottom: the matched filter's — every short line through one
compact overdensity at the edge of a mask (near the Fornax dwarf) is bright;
the line network gives that window 0.05. `run.py figures`.*

Neither is the network inventing lines: the Sagittarius wing is a real
linear overdensity, and the network is right to find it. Both windows reach
out of the calibration sky — at m−M 17 the line network's strongest have
only 59-69% of their valid sky on it — into the training-only sky, which is what the
calibration mask removes. The per-pixel false-alarm rates are measured on
calibration pixels alone; the window-level equivalent is a **stream-free
window lying on calibration sky** (at least 95% of its valid pixels). With
that rule the 1% levels at m−M 17 fall to 0.45 and 6.6, and that is the rule
used below.

## Results

```{image} ../figures/line_model/hough_blind_fold0.png
:alt: Copies found without the track, against input S/N, near and far: per-pixel network, line network, line network on S/N inputs, matched-filter lines
:width: 100%
```

```{image} ../figures/line_model/hough_known_fold0.png
:alt: Copies found along the known track, against input S/N, near and far: line network, line network on S/N inputs, matched-filter lines, matched-filter band
:width: 100%
```

*Top: without the track, at one false line per hundred stream-free windows
(the per-pixel network's test, which knows the track, for reference).
Bottom: along the known track. The DES 2018 copies at full brightness and 1
and 1.5 mag/arcsec² fainter, near (m−M < 16.5) and far; dashed, no network.
`run.py figures`.*

| half-recovery input S/N (DES 2018 copies found) | near | far |
|---|---|---|
| **without the track** | | |
| per-pixel network, 4 models — its test knows the track | 12.1 (72%) | 8.9 (93%) |
| line network, window-normalized inputs | 20.6 (55%) | 12.3 (92%) |
| **line network, S/N inputs** | **7.9 (88%)** | 12.8 (72%) |
| matched-filter line search | 12.5 (78%) | **10.0 (99%)** |
| per-pixel network + line search | 24.1 (32%) | 17.1 (57%) |
| **along the known track** | | |
| line network, window-normalized inputs | 7.7 | 8.6 |
| line network, S/N inputs | 6.0 | 8.9 |
| matched-filter lines | 6.2 | 4.7 |
| matched-filter band ({doc}`../real_des/labels_normalization`) | 5.2 | 5.0 |

Stream by stream, at full brightness (of 8 places, without the track):

| | input S/N | per-pixel | line network | line network, S/N inputs | matched-filter lines |
|---|---|---|---|---|---|
| Wambelong (m−M 15.9) | 8.6 | 1 | 0 | **6** | 0 |
| Jhelum (15.6, the widest) | 17.9 | 5 | 0 | 7 | 8 |
| distance scan, m−M 15 | 9.4 | 3 | 4 | **7** | 3 |
| Tucana III (17.0, 4.8° long) | 30.2 | 8 | 7 | **0** | 8 |
| Willka Yaku (17.7, 6.4° long) | 22.6 | 8 | 8 | **0** | 8 |

What it says:

1. **The integration works where it was missing.** On S/N inputs, the line
   network finds near, faint streams better than anything before it — half
   of them at an input S/N of 7.9 without the track, and Wambelong, which
   the per-pixel network found once in eight and the matched filter's lines
   never. Along a known track it reaches the matched filter's line sums.
2. **What reaches the lines must be an S/N.** The same architecture on
   per-window normalized maps is worse than the matched filter everywhere:
   a window's mean and spread are not its background and noise, so its line
   sums are not S/N, and the network has to learn what the matched filter is
   given.
3. **It misses the brightest short streams.** Tucana III and Willka Yaku,
   the two shortest copies (4.8° and 6.4°), at input S/N 30 and 23, are never
   found on S/N inputs, though every other method finds them — the far
   curve falls at the brightest copies. On S/N maps a short bright stream
   looks like the compact overdensities (clusters, dwarfs) of the stream-free
   sky that the network learned to ignore; a hypothesis, not tested.
4. **The matched filter's line search is a strong blind baseline**, at no
   cost: 99% of the distant copies and 78% of the near ones, at one false
   line per hundred stream-free windows. Any network has to beat it.
5. **The per-pixel network does not become a line detector by adding a line
   search on top**: its output, summed along lines, has a heavy-tailed null
   and is the weakest of all.

The S/N-input models' recorded training time (114 minutes) counts the
laptop asleep, lid closed, for most of it (the power log shows it); their
cost is that of the line network plus about 3 minutes for the normalizer
(36 ms a window, against 1 ms).

### What to try next

- **The short bright streams**: more short streams in training (the length
  is drawn from 4 to 30 degrees, uniformly), and check the hypothesis on
  the stream-free sky's compact overdensities.
- **The line network and the matched-filter line search together** — a copy
  found by either, at half the false-line rate each — since they fail on
  different streams. Both are configurations; the combination is one more.
- **The long tier** (19,200 windows) for the S/N-input model.
- ~~The real streams~~: done in {doc}`line_sky` — the line network over the
  DES sky with the fourteen DES 2018 streams in it, mapped.

## Reproducing

```bash
# the line models (two seeds each; ~17 min a model)
python scripts/experiments/line_model/run.py train --config hough/band --seed 42
python scripts/experiments/line_model/run.py train --config "hough/band residual" --seed 42
# window-level scoring: a line model, or a per-pixel ensemble with "lines"
# (~10 min each; --sets fainter for the fainter copies)
python scripts/experiments/line_model/run.py evaluate --config hough/band
python scripts/experiments/line_model/run.py evaluate --config "count/window x4 lines"
python scripts/experiments/line_model/run.py figures       # the figures of this page
```
