# The line model over the DES sky

{doc}`window_level` scored the line model on simulated copies of the DES
2018 streams. This page runs it over the **real DES sky, with the real
streams in it**, and draws where it finds lines, to compare with the
fourteen streams of DES 2018 (Shipp et al. 2018) and with the per-pixel
network of the first training ({doc}`results`, 5 of 14). The matched
filter's own line search is run alongside, and the two are combined; the
detected segments are joined into a catalogue of tracks. How the line model
works is in the guide, {doc}`../../narrative/line_model`.

Branch `line-model-sky`; code `scripts/experiments/real_des/patches.py`
(`line-sky`).

**In short.** Searching the whole sky without knowing any track, the line
network finds **six DES 2018 streams beyond chance** — ATLAS, Phoenix,
Indus, Jhelum, Chenab, Elqui — with a clean map: 138 lines, five in six
along a stream or around the Magellanic Clouds. The matched filter's line
search, once the bright dwarf galaxies and the globular clusters are masked
(every line through them is bright), finds **five** — ATLAS, Phoenix, Elqui,
and the two short, bright ones the line network misses, Tucana III and
Willka Yaku. **Together, each at half its false-alarm rate (about 1% of
stream-free windows for the two), they find seven**; at their own rates,
eight; the per-pixel network found five. Joined across windows and
distances, the lines make 54 tracks seen more than once: 13 along DES 2018
streams, 22 in the Magellanic Clouds' outskirts, 10 along other known
streams — among them the eastern extension of Jhelum that Ibata et al.
(2024) proposed — and 7 unidentified. The line network misses short streams
because its label taught it to: the band label's 4° minimum made every
window holding a shorter clear stretch a negative example.

## How the maps are made

- **The sky**: the DES Y6 inference sky — the training sky's stars with the
  known streams put back (4,600 deg² of valid sky) — with the same galaxy and
  artefact mask as every sky the models saw, and the bright dwarf galaxies
  and globular clusters masked (below).
- **The search**: the 164 half-overlapping 11° tiles covering it, each at
  the nine queried distances m−M 15, 15.5, …, 19.
- **Out of fold**: a tile whose centre is in fold 0 is scored by the two
  line models trained on fold 1, and the other way round, so no model
  judges sky it trained on. The fold-1 models (`hough/band residual`, two
  quick models) were trained for this page; scored on fold 0's copies they
  do what the fold-0 models do on fold 1 (below).
- **A detection**: a line that is a peak over (θ, ρ) — the highest within
  5 × 5 cells — and scores above the level that 1% of stream-free windows
  reach, measured at that distance on the calibration sky of the tile's
  fold. Its stretch over valid sky in the tile is a segment on the sky.
- **The matched filter's line search** runs on the same tiles: this sky's
  counts minus their smooth local background, over its square root, summed
  along every line, with its own 1% levels.

The fold-1 models on fold 0, against the fold-0 models on fold 1 (half-
recovery input S/N of the DES 2018 copies, near / far):

| | trained on fold 0, scored on fold 1 | trained on fold 1, scored on fold 0 |
|---|---|---|
| line network, without the track | 7.9 / 12.8 | 8.1 / 12.7 |
| line network, along the track | 6.0 / 8.9 | 7.7 / 7.5 |
| matched-filter lines, without the track | 12.5 / 10.0 | 7.7 / 8.1 |
| matched-filter lines, along the track | 6.2 / 4.7 | 5.2 / 3.6 |

The line network does the same on both halves of the sky; the matched
filter's line search does better on fold 0, where it matches the line
network on near streams too.

## Suppressing the bursts

Every line through a bright blob is bright (the sinusoid of the guide's toy
figure). On the first run, without masking anything more than the training
sky does, the matched filter drew 1,003 lines — 381 radiating from the
Fornax dwarf and 113 from Sculptor, whose stars reach past the training
mask's radius — and "found" Aliqa Uma, whose track passes Fornax, by these
bursts alone. The calibration sky already removes dwarfs to 12 half-light
radii and globular clusters to 2° (`run.py` `object_mask`); the search now
masks them too, but **only the dwarfs brighter than M_V = −8** (Fornax,
Sculptor and distant galaxies with small discs): the ultra-faint dwarfs
make no bursts, and some lie on streams — masking all of them hid the
middle of Tucana III (its own progenitor) and part of Indus (beside
Tucana II).

| | lines | around Fornax / Sculptor | DES 2018 streams found |
|---|---|---|---|
| matched filter, training mask only | 1,003 | 381 / 113 | 4, with Aliqa Uma by the Fornax burst |
| **matched filter, bright dwarfs and clusters masked** | **430** | **0 / 0** | **5**, Aliqa Uma gone, Phoenix and ATLAS now beyond chance |
| line network, training mask only | 152 | 0 / 0 | 6 |
| line network, bright dwarfs and clusters masked | 138 | 0 / 0 | 6 |

The line network learned to ignore the bursts and is unchanged.

## The maps

```{image} ../figures/real_des_patches/line_sky_network.png
:alt: Lines the line network detects over the DES sky, in three ranges of distance, with the DES 2018 tracks and the Magellanic Clouds and bright dwarfs marked
:width: 100%
```

```{image} ../figures/real_des_patches/line_sky_matched_filter.png
:alt: Lines the matched filter's line search detects over the DES sky with the bright dwarfs and clusters masked, in three ranges of distance
:width: 100%
```

```{image} ../figures/real_des_patches/line_sky_combined.png
:alt: Lines of both searches, each at half its false-alarm rate, over the DES sky
:width: 100%
```

*Every detected line, coloured by its queried distance, in three ranges of
distance; light grey, the inference sky; thick grey, the DES 2018 tracks of
the streams whose distance falls in the range; red crosses, the Magellanic
Clouds (their centres lie past the footprint's edge, drawn at it) and the
Fornax and Sculptor dwarfs. Top: the line network at its 1% level; middle:
the matched filter's line search at its 1% level, dwarfs and clusters
masked; bottom: both, each at half its rate (solid: line network; dashed:
matched filter). `patches.py line-sky`.*

Where the lines are (a line counts as a stream's if at least 3° of it runs
within max(1°, two widths) of its track at any distance; as an object's if
it passes within 20° of the LMC or 12° of the SMC, the calibration's discs):

| | line network | matched-filter line search | both, half rate each |
|---|---|---|---|
| along a DES 2018 track | 74 | 89 | 141 |
| around the LMC | 43 | 165 | 187 |
| around the SMC | 4 | 56 | 56 |
| elsewhere | 17 | 120 | 109 |
| **all** | **138** | **430** | **493** |

- **The Magellanic Clouds' outskirts** are real stars at their distance
  (the LMC at m−M 18.5, the SMC at 19): the line network draws them mostly
  at m−M 18-19, the matched filter at every distance.
- **Elsewhere**, the line network's 17 lines are mostly a group in the east
  (RA 73-94°, Dec −23° to −46°, m−M 15-16.5) — part of it along known
  streams (below). The matched filter's include a bundle along Dec −41°,
  RA 300-333°, at m−M 16-18.5, parallel to the footprint's northern edge
  1.5-3° inside it, which the line network does not see.

## The fourteen DES 2018 streams

```{image} ../figures/real_des_patches/line_sky_streams.png
:alt: The fourteen DES 2018 streams, nearest first, and which method finds them
:width: 85%
```

A stream is found by a line search if a detected line at the queried
distance nearest its own runs along its track (at least 3° within
max(1°, two widths) of it). A long, wide track is crossed by a stray line
often — a random track of Jhelum's shape is, in 42% of places — so what
tells a stream from chance is **how many** lines run along it. Each stream's
track was placed at 200 random places and position angles on the footprint;
*p* is the share of them with at least as many lines along it as the stream
itself, and a stream counts as found when *p* ≤ 0.05. The per-pixel
network's test (a band's flagged pixels against 200 null bands, {doc}`results`)
is already one of significance.

The **combination** keeps either search's lines above its level at half the
false-alarm rate (0.5% of stream-free windows each); on the same
stream-free windows, the two together give a line in 0.83-1.00% of them,
the rate of each search alone. **Either** — a stream found by either
search at its own 1% level — is shown too, at about twice that rate.

| | m−M | per-pixel network | line network: lines (*p*) | matched-filter lines (*p*) | both, half rate (*p*) |
|---|---|---|---|---|---|
| Jhelum | 15.6 | | **5 (0.035)** | | **5 (0.030)** |
| Wambelong | 15.9 | | | | |
| Indus | 16.1 | | **5 (0.015)** | | **5 (0.040)** |
| Turbio | 16.1 | | | | |
| Phoenix | 16.4 | | **2 (0.030)** | **2 (0.030)** | **3 (0.040)** |
| ATLAS | 16.8 | **found** | **4 (0.005)** | **6 (0.015)** | **10 (0.005)** |
| Molonglo | 16.8 | | | | |
| Ravi | 16.8 | | | 1 (0.20) | |
| Tucana III | 17.0 | **found** | | **4 (0.005)** | **3 (0.010)** |
| Turranburra | 17.2 | | 1 (0.080) | | |
| Aliqa Uma | 17.3 | | | | |
| Willka Yaku | 17.7 | **found** | | **5 (0.015)** | **4 (0.020)** |
| Chenab | 18.0 | **found** | **4 (0.005)** | 2 (0.16) | 5 (0.080) |
| Elqui | 18.5 | **found** | **2 (0.010)** | **6 (0.010)** | **5 (0.015)** |
| **found** | | **5** | **6** | **5** | **7** (either at its own rate: 8) |

What it says:

1. **The two line searches are complementary**: the line network finds the
   near, wide streams — Jhelum, Indus — that neither the matched filter nor
   the per-pixel network finds; the matched filter finds the short, bright
   ones — Tucana III, Willka Yaku — that the line network misses.
2. **Combined at the same false-alarm rate, they find seven of the
   fourteen**, against five for the per-pixel network; at their own rates,
   eight. Chenab, found by the line network alone, falls just short in the
   combination (*p* = 0.08): at half its rate, the line network keeps fewer
   of its lines, and the matched filter's add more chance ones.
3. **Five are found by nothing**: Wambelong, Turbio, Molonglo, Ravi,
   Aliqa Uma; {doc}`des2018_reproduction` found Ravi and Molonglo not to be
   in the data at all.

## Tracks: a catalogue of candidates

The detections of the combination are joined into tracks: two segments are
one track's if their great circles agree within 5°, they come within 1° of
each other, and their queried distances differ by at most 0.5 — the same
structure seen by overlapping tiles and neighbouring distances. Each track
gets the great circle through its segments, its extent along it, its mean
distance (weighted by how far each segment stands above its level), and an
identification, in this order: a DES 2018 stream at its distance (within
1 mag); the outskirts of a Magellanic Cloud (its middle within the
calibration's 20° or 12°); the footprint's edge (60% of it within 1.5° of
sky outside the footprint); another stream of galstreams (3° of it within
1.5° of the stream's track); or none.

```{image} ../figures/real_des_patches/line_sky_tracks.png
:alt: The tracks seen in at least two segments over the DES sky, coloured by distance, with the DES 2018 tracks underneath
:width: 100%
```

*The 54 tracks seen in at least two segments (of 94), coloured by their
mean distance, thicker for more segments; the unidentified numbered
(line_sky/tracks_hough_band residual.csv). `patches.py line-sky`.*

| | tracks seen in ≥ 2 segments |
|---|---|
| along a DES 2018 stream, at its distance | 13 — ATLAS (one track of 25.7°, from 37 segments in 9 windows), Jhelum, Indus, Chenab, Elqui, Phoenix, Willka Yaku, Tucana III |
| the Magellanic Clouds' outskirts | 22 |
| along another known stream | 10 |
| unidentified | 7 |
| the footprint's edge; a DES 2018 track at another distance | 1; 1 |

Among the other known streams:

- **Jhelum's eastern extension** (Ibata et al. 2024), at RA 77°, Dec −29°:
  found by the line network, at m−M 15.2 — Jhelum's own distance (15.6 in
  DES 2018).
- **New-4** (Ibata et al. 2024), at RA 92°, Dec −40°, m−M 16.3, by the
  matched filter; beside it, two unidentified tracks of the line network at
  m−M 15.8-16 (RA 91-94°, Dec −40° to −45°) — perhaps the same structure.
- **M2's stream** (Grillmair 2022), along the narrow strip of the footprint
  at Dec 0°, m−M 16.2 — at a strip's edge, where lines run easily.
- The bundle along Dec −41° (RA 300-333°), found by the matched filter
  alone, falls on the tracks of Orphan-Chenab, C-7 and Phlegethon by
  position, but at distances that do not fit them, parallel to the
  footprint's edge: more likely an artefact of the matched filter's
  background there than a stream.

The catalogue is a list of candidates to look at, not a detection
significance: a track's segments are not independent (overlapping windows,
neighbouring distances), and known streams are matched by position.

## Why the line network misses short streams

On the copies, the line network finds Tucana III (4.8° long) at 6% of its
places and Willka Yaku (6.4°) at 44%, at input S/N 29 and 24, where long
copies at S/N 10 are found. Making them fainter does not change it — it is
the length. A **length scan** settles it: one stream at m−M 17 and a width
of 0.2°, at six lengths, bright (32 mag/arcsec²) and moderate (33.5), 8
places each, scored as in {doc}`window_level` (fold-0 models on fold 1):

| length | input S/N (bright / moderate) | line network, without the track (bright / moderate) | matched-filter lines (bright / moderate) |
|---|---|---|---|
| 4° | 27 / 7 | 0% / 0% | 100% / 12% |
| 5° | 31 / 8 | 12% / 12% | 100% / 12% |
| 6° | 34 / 9 | 0% / 12% | 100% / 38% |
| 8° | 39 / 10 | 38% / 62% | 100% / 88% |
| 10° | 46 / 12 | 88% / 50% | 100% / 100% |
| 15° | 56 / 14 | 100% / 88% | 100% / 100% |

A 4° stream at input S/N 27 — plain to see — is never found: the network
answers "no line" to short streams however bright. The reason is in its
label: the band label marks a stream only where at least **4°** of it is
visible in the window at the queried distance (the minimum chosen so the
per-pixel network would not learn blobs). Every training window holding a
shorter clear stretch — a stream cut by the window's edge, or partly masked
— was therefore a *negative* example, and the line model learned to ignore
short segments.

<!-- the test of a 2-degree minimum is added below -->

## Reproducing

```bash
# the fold-1 line models, and their 1% levels on fold 0 (~20 + 10 min)
python scripts/experiments/real_des/patches.py train --config "hough/band residual" --seed 42 --train-sky fold1
python scripts/experiments/real_des/patches.py train --config "hough/band residual" --seed 43 --train-sky fold1
python scripts/experiments/real_des/patches.py evaluate --config "hough/band residual" --train-sky fold1
python scripts/experiments/real_des/patches.py evaluate --config "hough/band residual" --train-sky fold0
# the whole sky: detections, matches, chance, maps, tracks (~10 min)
python scripts/experiments/real_des/patches.py line-sky
# the length scan
python scripts/experiments/real_des/patches.py evaluate --config "hough/band residual" --sets "length scan"
```
