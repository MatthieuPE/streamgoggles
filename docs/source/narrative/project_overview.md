# Project overview: finding streams without looking

Stellar streams are found today by eye, in matched-filter maps. This project
trains a network on simulated streams to do that search instead. The best
model finds **8 of the 14 DES 2018 streams without being told where they
are**, and 9 with the matched filter's own line search beside it
({doc}`../experiments/line_model/summary`).

This page tells the story from the start, for a reader who knows neither
streams nor machine learning in depth. It then gives the details of the line
model: what goes in, what comes out, how the length of a line is dealt with,
and how lines found in separate windows are combined over the sky. The
figures are toy illustrations, not survey data; they are made by
`scripts/overview/figures.py`.

## 1. Stellar streams

```{image} figures/overview/1_stream_formation.png
:alt: A satellite on its orbit around the Milky Way, stretched into a leading and a trailing tail of stars
:width: 45%
```

- A globular cluster or a dwarf galaxy orbits the Milky Way.
- The Galaxy's tides pull stars off it. They spread along the orbit, ahead and
  behind.
- The result is a **stream**: long (tens of degrees), thin (0.1-1.5° in our
  training set), and very faint.
- Streams trace the Milky Way's mass and its dark matter (review: Bonaca &
  Price-Whelan 2025, [arXiv:2405.19410](https://arxiv.org/abs/2405.19410)).

A stream adds a few stars per pixel to a background of many Milky Way stars.
It cannot be seen in a plain star-count map.

## 2. How streams are found: the matched filter

```{image} figures/overview/2_matched_filter.png
:alt: All stars show nothing; a cut in color and brightness keeps the stream's stars; the filtered map shows the stream
:width: 100%
```

- A survey gives, for each star, a position, a color and a brightness.
- The stars of one stream share one age, one metallicity and one distance.
  In color and brightness they lie on one track, an isochrone.
- Keeping only the stars near that track removes most of the background.
  Counting them per sky pixel gives a **matched-filter map**, where the
  stream stands out (Rockosi et al. 2002, AJ 124, 349).
- The distance is unknown, so the map is made at a series of **trial
  distances**.

How the filters are built here: {doc}`data_generation`.

## 3. The bottleneck: inspection by eye

```{image} figures/overview/3_bottleneck.png
:alt: A grid of eighteen noisy maps, sky regions by distance; one holds a faint stream
:width: 100%
```

- The 14 DES 2018 streams, 11 of them new, were found by inspecting such
  maps (Shipp et al. 2018, ApJ 862, 114,
  [arXiv:1801.03097](https://arxiv.org/abs/1801.03097)).
- One survey gives many maps: sky regions × trial distances.
- A visual search has no measured completeness and no false-alarm rate.
- LSST will cover about 18,000 deg², deeper (Ivezić et al. 2019, ApJ 873,
  111). Inspecting every map by eye will not scale.

## 4. The approach: simulations as a training set

```{image} figures/overview/5_ml_pipeline.png
:alt: Simulate streams, inject them into the real sky, train the network, scan the whole survey, candidate streams
:width: 100%
```

- **Simulate** streams with `streamobs`: distance m−M 15-19, surface
  brightness 32-34.5 mag/arcsec², width 0.1-1.5°, length 4-30°, age
  9-13.5 Gyr.
- **Inject** them into the real DES Y6 sky, with the known streams,
  clusters and dwarfs masked ({doc}`real_des_background`,
  {doc}`../experiments/real_des/index`).
- Each training example comes with its answer: we know where we put the
  stream. Real streams are too few to learn from; simulated ones are
  unlimited.
- The sky is cut into two folds of 20° stripes. Models train on one fold and
  are judged on the other.

## 5. The model: maps in, lines out

```{image} figures/overview/7_model.png
:alt: Input maps of one window, a U-Net, sums along every line, a small network over the lines, one probability per line
:width: 100%
```

The network answers one question for every straight line through a sky
window: **does a stream lie along it?** Full description:
{doc}`line_model`; code: {py:class}`~streamgoggles.models.hough.HoughUNet`.

| step | what it does | shape |
|---|---|---|
| window | an 11° gnomonic projection of the sky, 96 × 96 pixels of 0.115°; great circles are straight lines | |
| input | matched-filter counts at the queried distance d and at d ± 0.5 mag, an off-track color box (background only), and 3 constant maps holding the distances, so one network serves every distance | 7 × 96 × 96 |
| S/N maps | each count map becomes its excess over a local background plane, over the plane's square root (`ResidualNormalizer`) | 7 × 96 × 96 |
| U-Net backbone | depth 2, base width 12; 8 feature maps, no activation, no per-pixel label | 8 × 96 × 96 |
| + the input | the 7 inputs are appended untouched, so the matched filter's own line sums always reach the end | 15 × 96 × 96 |
| line layer | every channel summed along 90 angles × 69 offsets (a Hough transform), divided by √n; a fixed sparse matrix, not trained, gradients flow through it | 15 × 90 × 69 |
| over the lines | two 3 × 5 convolutions over (angle, offset), batch norm, ReLU | 16 × 90 × 69 |
| head | a 1 × 1 convolution and a sigmoid: one probability per line | 1 × 90 × 69 |

Only the U-Net and the small network over the lines are trained. The line
layer is fixed: it is the integration along the stream, written into the
network.

## 6. Why lines

```{image} figures/overview/4_line_sums.png
:alt: A faint line in noise, invisible by eye, and the sums along every line, where it stands as one peak
:width: 80%
```

- A stream is faint at every pixel, but long. Its evidence is in the **sum
  along its track**.
- In the toy above, the stream adds 0.6σ per pixel. Summed along its line it
  stands at about 7σ.
- The first network answered pixel by pixel (a U-Net with a per-pixel
  label). 90% of what decides a pixel lies within 1.5° of it. It acted as
  a local detector: 5 of the 14 DES 2018 streams, even along their known tracks
  ({doc}`../experiments/real_des/results`).
- No change of that network closed the gap: depth, loss, label, fainter
  training ({doc}`../experiments/real_des/labels_normalization`). Building
  the line sum into the model did ({doc}`../experiments/line_model/index`).

## 7. Training

- **Label.** The stream's band (one width on each side of its track) at the
  queried distance, kept where at least **2°** of it stands at **S/N ≥ 5**
  in the window (`band2s5`; `scripts/experiments/real_des/patches.py`). The
  band is turned into lines (`hough_target`): the lines holding at least 80%
  of the best line's band pixels, and their neighbours one step away in angle
  and offset, are positive.
- **Empty windows.** 30% of training windows hold no stream. A window whose
  stream is not visible at the queried distance has no positive line.
- **Loss.** Binary cross-entropy over the valid lines. The head's bias starts
  at the share of positive lines.
- **Budget.** 4,800 windows per model (about 35 min). Four models per fold,
  their probabilities averaged (`hough/band2s5 residual x4`).

## 8. How the length of a line is dealt with

```{image} figures/overview/8_line_length.png
:alt: Idealized line sum against the stream's length inside the window, for window lines and sub-window lines, with the label's minimum length marked
:width: 70%
```

*Idealized: a stream adding 0.6σ per pixel along one line, no network.
Window lines about 96 pixels long, sub-window lines 48.*

Several lengths meet in one window: the length of each line, the length of
the stream inside the window, and the stream's full length on the sky.

1. **Lines of different lengths.** A line through the window's centre is
   about 96 pixels long; one across a corner is shorter. Each line sum is
   divided by √n, its number of pixels. On white noise every line sum then
   has unit variance, whatever its length. On an S/N map, the sum *is* the
   line's matched-filter S/N.
2. **Very short lines are left out.** Lines with fewer than 20 pixels
   (2.3°) in the window, across its corners, are left out of the loss and of
   the scoring (`HoughLines.valid`).
3. **A stream shorter than the window is diluted.** Its signal grows with
   its own length L, the noise with √(line length). A 4° stream on an 11°
   line keeps about √(4/11) ≈ 60% of the S/N a 4° line would give it (figure
   above).
4. **The label decides which short streams count.** With a 4° minimum,
   every window holding a shorter clear stretch was a negative example, and
   the network learned to ignore short streams: a bright 4-6° stream was
   never found. With a 2° minimum it is found (length scan,
   {doc}`../experiments/line_model/line_sky`):

   | stream length (bright, 32 mag/arcsec²) | 4° | 5° | 6° | 8° | 10° | 15° |
   |---|---|---|---|---|---|---|
   | found without the track, 4° label | 0% | 12% | 0% | 38% | 88% | 100% |
   | found without the track, **2° label** | 25% | 75% | 100% | 100% | 100% | 100% |

5. **Sub-window ("segment") lines** were tried against dilution: the lines
   of nine half-overlapping 5.5° sub-windows beside the window's own
   (`SegmentLines`). They helped the matched filter's line search (6 → 8
   streams on the sky), not the network: four segment models find what four
   window-line models find ({doc}`../experiments/line_model/des2018_known`).
   The adopted model uses window lines only.
6. **Width.** A wide stream spreads over neighbouring offsets. The two
   convolutions over (angle, offset) span about 2° of offset and gather it
   back.
7. **A stream longer than the window** is seen in pieces, one per window it
   crosses. A curved track is approximated, in each window, by its best
   straight line. Section 9 joins the pieces.

## 9. From windows to the sky: combining lines

```{image} figures/overview/9_windows_to_tracks.png
:alt: Half-overlapping windows each report a straight segment of a curved stream; segments with close directions are joined into tracks
:width: 100%
```

*Toy flat sky. A curved stream crosses 14 windows; their segments join into
tracks where neighbouring directions agree within 5°. Here a strongly curved
stream is cut into several tracks.*

The windows are scored independently. Probabilities are **not** stitched
into a sky map. Lines are combined as **segments on the sky**, in five
steps (`run.py line-sky`, `line_sky_tracks`;
{doc}`../experiments/line_model/line_sky`):

1. **Tiles.** 164 half-overlapping 11° tiles cover the 4,600 deg² of the
   inference sky. Each is scored at the 9 queried distances, m−M 15 to 19 by
   0.5.
2. **Out of fold.** A tile whose centre lies in one fold is scored by the
   models trained on the other. No model judges sky it trained on.
3. **Detections.** A line is detected when it is a local peak over
   (angle, offset), the highest within 5 × 5 cells, and scores above a
   **level**. The level is set on stream-free windows of the calibration
   sky, per fold and per distance: 1% of those windows hold a line above it.
   Each detection becomes a **segment**: the stretch of its line over valid
   sky in the tile, a great-circle arc.
4. **Two searches together.** The network and the matched filter's own line
   sums run on the same tiles. Each keeps its lines above its 0.5% level.
   Together, 0.83-1.00% of stream-free windows hold a line: the rate of one
   search alone.
5. **Tracks.** Segments are joined (union-find) when their great circles'
   poles agree within 5°, they come within 1° of each other, and their
   queried distances differ by at most 0.5 mag. Overlapping tiles and
   neighbouring distances see the same structure several times: ATLAS gave
   one 25.7° track from 37 segments in 9 windows. Each track gets the great
   circle of least scatter through its segments, its extent, a mean distance
   weighted by how far each segment stands above its level, and an
   identification.

**Is a known stream found?** At the queried distance nearest its own, count
the segments that run along its track: at least 3° within max(1°, two
widths) of it, and within 15° of its direction. A long, wide track catches a
stray line now and then, so the count is compared with the same track placed
at 200 random places and position angles. The stream is found when
*p* ≤ 0.05.

**Limits of the combination.**

- The catalogue lists candidates, not significances. A track's segments are
  not independent (overlapping tiles, neighbouring distances). A discovery
  claim needs a look-elsewhere correction over the whole search.
- One great circle per track. A curved stream can be cut into several
  tracks. The leads that held were refitted afterwards with a curved track
  ({doc}`../experiments/line_model/long_and_fits`).
- Every straight feature makes lines: mask edges with stars beyond them,
  the Magellanic Clouds' outskirts, bright dwarfs. Half the unexplained lines
  run along one edge of the valid sky, at Dec −40°
  ({doc}`../experiments/line_model/summary`).

## 10. Result

```{image} figures/overview/6_scorecard.png
:alt: The fourteen DES 2018 streams: 5 found by the pixel network along their tracks, 9 by the line searches without their tracks
:width: 100%
```

| | along the known track | without the track |
|---|---|---|
| per-pixel network (first training) | 5 | — |
| matched filter's line sums | 9 | 6 |
| **line network** | **11** | **8** |
| both line searches, each at half its rate | 10 | 8 (sky search: **9**) |

*Of the 14 DES 2018 streams. Ravi and Molonglo are not in our data
({doc}`../experiments/real_des/des2018_reproduction`). Source:
{doc}`../experiments/line_model/summary`.*

- Trained only on simulations, the line search recovers most known streams
  with no human in the loop.
- The real streams are found as often as their simulated copies at the same
  strength predict: the simulations are a sound test bench.
- Three tracks beyond DES 2018 held up: Tucana III's eastern extension,
  NGC 1261's stream, and a stream along Leiptr's direction.

## Reproducing the figures

```bash
python scripts/overview/figures.py   # writes docs/source/narrative/figures/overview/
```

The figures are toy data. Their numbers (6,210 lines, 11° windows, 5° and
1° joining rules, the scorecard) follow the code and the experiment pages
linked above.

## References

- Bonaca, A. & Price-Whelan, A. M. (2025), Stellar streams in the Gaia era,
  [arXiv:2405.19410](https://arxiv.org/abs/2405.19410)
- Duda, R. O. & Hart, P. E. (1972), Use of the Hough transformation to detect
  lines and curves in pictures, Commun. ACM 15, 11
- Ivezić, Ž. et al. (2019), LSST: from science drivers to reference design
  and anticipated data products, ApJ 873, 111
- Rockosi, C. M. et al. (2002), A matched-filter analysis of the tidal tails
  of the globular cluster Palomar 5, AJ 124, 349
- Ronneberger, O., Fischer, P. & Brox, T. (2015), U-Net,
  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Shipp, N. et al. (2018), Stellar streams discovered in the Dark Energy
  Survey, ApJ 862, 114, [arXiv:1801.03097](https://arxiv.org/abs/1801.03097)
