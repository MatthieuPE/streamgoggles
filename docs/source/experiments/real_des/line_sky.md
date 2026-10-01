# The line model over the DES sky

{doc}`window_level` scored the line model on simulated copies of the DES
2018 streams. This page runs it over the **real DES sky, with the real
streams in it**, and draws where it finds lines, to compare with the
fourteen streams of DES 2018 (Shipp et al. 2018) and with the per-pixel
network of the first training ({doc}`results`, 5 of 14). The matched
filter's own line search is run alongside. How the line model works is in
the guide, {doc}`../../narrative/line_model`.

Branch `line-model-sky`; code `scripts/experiments/real_des/patches.py`
(`line-sky`).

**In short.** Searching the whole sky without knowing any track, the line
network draws 152 lines at nine distances: half of them follow DES 2018
streams, a third the outskirts of the Magellanic Clouds. It finds **six DES
2018 streams beyond chance** — ATLAS, Phoenix, Indus, Jhelum, Chenab, Elqui
— three of them (Phoenix, Indus, Jhelum) found by neither the per-pixel
network nor the matched filter. The matched filter's line search finds four
beyond chance — Tucana III, Elqui, Willka Yaku, Aliqa Uma — but draws 1,003
lines, most of them bursts around the Fornax and Sculptor dwarfs and the
Magellanic Clouds. **Together, the two line searches find nine of the
fourteen streams**; the per-pixel network found five.

## How the maps are made

- **The sky**: the DES Y6 inference sky — the training sky's stars with the
  known streams put back (4,600 deg² of valid sky) — with the same galaxy and
  artefact mask as every sky the models saw.
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

## The maps

```{image} ../figures/real_des_patches/line_sky_network.png
:alt: Lines the line network detects over the DES sky, in three ranges of distance, with the DES 2018 tracks and the Magellanic Clouds and bright dwarfs marked
:width: 100%
```

```{image} ../figures/real_des_patches/line_sky_matched_filter.png
:alt: Lines the matched filter's line search detects over the DES sky, in three ranges of distance
:width: 100%
```

*Every line above its 1% level, coloured by its queried distance, in three
ranges of distance; light grey, the inference sky; thick grey, the DES 2018
tracks of the streams whose distance falls in the range; red crosses, the
Magellanic Clouds (their centres lie past the footprint's edge, drawn at it)
and the Fornax and Sculptor dwarfs. Top: the line network; bottom: the
matched filter's line search. `patches.py line-sky`.*

Where the lines are (a line counts as a stream's if at least 3° of it runs
within max(1°, two widths) of its track at any distance; as an object's if
it passes within 20° of the LMC, 10° of the SMC, 4° of a dwarf):

| | line network | matched-filter line search |
|---|---|---|
| along a DES 2018 track | 79 | 130 |
| around the LMC | 46 | 186 |
| around the SMC | 3 | 54 |
| around the Fornax dwarf | 0 | 381 |
| around the Sculptor dwarf | 0 | 113 |
| elsewhere | 24 | 139 |
| **all** | **152** | **1,003** |

- **The Magellanic Clouds' outskirts** are real stars at their distance
  (the LMC at m−M 18.5): the line network draws them mostly at m−M 18-19,
  the matched filter at every distance.
- **The bright dwarfs**: every line through a bright blob is bright (the
  sinusoid of the guide's toy figure), and the stars of Fornax and Sculptor
  past their masks make bursts of hundreds of lines for the matched filter;
  the line network learned to ignore them — the same reason it misses the
  short, bright streams.
- **Elsewhere**: the line network's 24 are mostly one group in the east, at
  RA 73-94°, Dec −23° to −46°, m−M 15-16.5, toward the Galactic anticentre
  — perhaps structure of the outer disc there, not identified here — and two
  lines along the narrow strip of the footprint at Dec 0°, an edge effect.

## The fourteen DES 2018 streams

```{image} ../figures/real_des_patches/line_sky_streams.png
:alt: The fourteen DES 2018 streams, nearest first, and which method finds them: per-pixel network 5, line network 6, matched-filter line search 4, either line search 9
:width: 75%
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

| | m−M | per-pixel network | line network: lines (*p*) | matched-filter lines: lines (*p*) |
|---|---|---|---|---|
| Jhelum | 15.6 | | **5 (0.045)** | |
| Wambelong | 15.9 | | | |
| Indus | 16.1 | | **5 (0.015)** | |
| Turbio | 16.1 | | | |
| Phoenix | 16.4 | | **2 (0.030)** | 2 (0.095) |
| ATLAS | 16.8 | **found** | **4 (0.005)** | 7 (0.060) |
| Molonglo | 16.8 | | | |
| Ravi | 16.8 | | | 1 (0.30) |
| Tucana III | 17.0 | **found** | | **4 (0.015)** |
| Turranburra | 17.2 | | 1 (0.080) | |
| Aliqa Uma | 17.3 | | | **4 (0.025)** |
| Willka Yaku | 17.7 | **found** | | **5 (0.020)** |
| Chenab | 18.0 | **found** | **4 (0.005)** | 2 (0.39) |
| Elqui | 18.5 | **found** | **3 (0.010)** | **7 (0.020)** |
| **found** | | **5** | **6** | **4** |

What it says:

1. **On the real sky the line network finds the near, wide streams** —
   Jhelum, Indus, Phoenix — which neither the per-pixel network nor the
   matched filter finds, with a clean map: five lines in six are along a
   stream or around the Magellanic Clouds.
2. **The matched filter's line search finds the short, bright ones** —
   Tucana III, Willka Yaku — which the line network misses, but its map is
   dominated by bursts around the dwarfs and the Clouds.
3. **The two are complementary**: together they find nine of the fourteen;
   the five left are Wambelong, Turbio, Molonglo, Ravi and Turranburra.
   Of these, {doc}`des2018_reproduction` found Ravi and Molonglo not to be in
   the data at all.

Caveats:

- *p* comes from random tracks over the whole footprint; near Fornax or the
  LMC the density of the matched filter's lines is far higher than its
  average, so a stream there — Aliqa Uma passes Fornax — is found by chance
  more often than *p* says.
- The windows are 11° across: a stream longer than that is found in pieces,
  and a line runs to the window's edges, so a segment can be longer than
  the stream it follows.
- The per-pixel network is the first training's (six models of
  {doc}`first_training`), not the four quick count-label models of
  {doc}`labels_normalization`, and its test is per band, not per line.

### What to try next

- **Suppress the blobs in the matched filter's line search** — mask the
  dwarfs' outskirts, or ask that a line's excess be spread along it rather
  than gathered in one place — and combine it with the line network.
- **Train the line network on short, bright streams**, which it misses.
- **Join segments across windows and distances into tracks**: a catalogue
  of candidates, rather than lines per window; and look at the group toward
  the anticentre.

## Reproducing

```bash
# the fold-1 line models, and their 1% levels on fold 0 (~20 + 10 min)
python scripts/experiments/real_des/patches.py train --config "hough/band residual" --seed 42 --train-sky fold1
python scripts/experiments/real_des/patches.py train --config "hough/band residual" --seed 43 --train-sky fold1
python scripts/experiments/real_des/patches.py evaluate --config "hough/band residual" --train-sky fold1
# the whole sky: detections, matches, chance, maps (~5 min, the sky built once)
python scripts/experiments/real_des/patches.py line-sky
```
