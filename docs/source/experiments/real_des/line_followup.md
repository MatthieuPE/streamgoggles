# Following up the line model: more models, and the leads

{doc}`line_sky` searched the real DES sky with the line model and left two
questions: does the 2° line model gain from more models or longer
training — enough to find Jhelum and Phoenix again — and are the tracks it
found outside the DES 2018 streams real? This page answers both.

Branch `line-model-followup`; code `scripts/experiments/real_des/patches.py`
(`lead_inspection`, and the configurations `hough/band2 residual x4`,
`s42` and `long`).

**In short.** More models help up to two per fold, longer training barely
— on the copies. On the real sky, **four quick models** give the line
network seven DES 2018 streams by itself (Jhelum, Indus, Turranburra, and
Tucana III among them) and **one long model** eight (Phoenix comes back);
with the matched filter's line search, they find **eight and nine of the
fourteen** — every stream but Wambelong, Turbio, Molonglo, Ravi and Aliqa
Uma, two of which are not in the data at all. Looked at one by one, the
leads of the catalogue split cleanly: **Leiptr, NGC 1261's stream and
Tucana III's eastern extension hold** — each runs along its proposed track
and shows a main sequence in the Hess difference, Tucana III's at its own
distance — while the eastern tracks are the footprint's edge, and the
"Jhelum extension" and "New-4" were lines crossing those streams, which the
matching now refuses.

## More models, longer training

The 2° line model of {doc}`line_sky` is two quick models (4,800 windows)
per fold. Two more seeds were trained on each fold (`hough/band2 residual
x4`: four models averaged), and one model per fold four times longer
(19,200 windows; `hough/band2 residual long`), compared with one quick
model (`hough/band2 residual s42`, the same seed). On the copies (fold-0
models on fold 1; half-recovery input S/N, without and along the track,
and the DES 2018 copies found at full brightness):

| 2° line model | without the track, near / far | along the track, near / far | DES 2018 copies found, near / far |
|---|---|---|---|
| 1 quick model | 8.7 / 12.9 | 6.6 / 11.5 | 88% / 79% |
| 2 quick models | 8.3 / 12.0 | 7.1 / 8.6 | 88% / 89% |
| 4 quick models | 8.3 / 12.2 | 7.1 / 8.5 | 90% / 82% |
| 1 long model (19,200 windows) | 8.1 / 12.4 | 7.4 / 12.0 | 85% / 71% |

**From one model to two, the limit improves; from two to four, it does
not** — as for the per-pixel network ({doc}`labels_normalization`). **Four
times the training does about what a second model does** for near streams
(8.1 against 8.7 for one quick model) and loses the short distant ones
again (far copies found 71% against 79%). Four
models find Jhelum's copies everywhere (100%, against 88%), but the short
streams' less often (Tucana III 25% against 50%, Willka Yaku 50% against
88%): averaging four models dilutes a response to short streams that only
some of them give.

On the sky (out of fold, as in {doc}`line_sky`):

| DES 2018 streams found beyond chance | line network | both searches, half rate each | either, its own rate |
|---|---|---|---|
| 4° label, 2 models | 7: ATLAS, Phoenix, Indus, Jhelum, Chenab, Elqui, Turranburra | 8 | 9 |
| 2° label, 2 models | 6: ATLAS, Indus, Jhelum, Chenab, Elqui, Tucana III | 8 | 8 |
| **2° label, 4 models** | **7**: ATLAS, Indus, Jhelum, Chenab, Elqui, Tucana III, Turranburra | **8** | **9** |
| 2° label, 1 long model (19,200 windows) | **8**: ATLAS, Indus, Jhelum, Chenab, Elqui, Tucana III, Phoenix and Turranburra (one line each) | **9** | **9** |

With four models **Jhelum is found with five lines, Turranburra comes back,
and Tucana III stays**: the line network alone finds seven, the same
number as the 4° label but with Tucana III for Phoenix. The matched filter
finds Phoenix, so the combination of the two searches finds the same eight:
ATLAS, Chenab, Elqui, Indus, Jhelum, Phoenix, Tucana III, Willka Yaku.

**The long model finds the most on the sky**: eight by the line network
alone, Phoenix and Turranburra by a single line each, and **nine with the
matched filter** — every DES 2018 stream but Wambelong, Turbio, Molonglo,
Ravi and Aliqa Uma. But it is one model per fold, and on the copies it has
lost what the 2° label gave: the length scan finds bright streams of 4-6°
at 0-12% again (the two quick 2° models: 25-100%), Tucana III's and Willka
Yaku's copies at 0%. A single model per fold is a noisy measure — its sky
result could be a good draw — and the long training may also learn the
short segments away again; four long models would tell.

## The leads, one by one

Each lead of the catalogue — a track found by a line search away from the
DES 2018 streams and the Magellanic Clouds — is looked at three ways, with
ATLAS, a DES 2018 stream both searches find, as the control:

- **a map**: the matched filter's excess over its smooth local background,
  in S/N, smoothed to 0.3°, around the lead at its distance, with the lead
  drawn as two rails 1° either side (so as not to hide it) and the track of
  the known stream it was matched to;
- **a distance profile**: the matched filter's excess in a band ±0.4° about
  the lead's track, against the same band at 200 random places of the
  calibration sky, at each queried distance — a stream peaks at its own;
- **a Hess difference**: the stars of the band minus those of flanking
  bands 1.5-3° either side, scaled by valid area, in colour and magnitude,
  with the matched filter's polygon at the lead's distance — a stream shows
  a main sequence there, a change of depth or of galaxy contamination a
  broad excess at the faintest magnitudes, a density gradient a broad excess
  everywhere.

And each is measured against the known stream it was matched to: the angle
between the two tracks where they meet, and how much of the lead lies
within 1.5° of the known track. This showed that the catalogue matched
crossing lines: a lead was given to a known stream if 3° of it lay within
1.5° of that stream's track, which a line crossing it at a large angle
satisfies. Lines must now run *along* a track (directions within 15°), for
the DES 2018 streams as for the others ({doc}`line_sky` uses this rule).

```{image} ../figures/real_des_patches/lead_atlas__control.png
:alt: ATLAS, the control: a streak on the map, a peak at its distance, a main sequence in the Hess difference
:width: 100%
```

*The control, ATLAS: the stream between the rails; a single peak of the
band's S/N at m−M 16.5-17 (9.4 against the null bands); a main sequence
from its turnoff at g ≈ 20.8 inside the filter's polygon. What a stream
looks like here.*

```{image} ../figures/real_des_patches/lead_leiptr.png
:alt: Leiptr: a main sequence in the Hess difference, the band's S/N highest at the nearest distance
:width: 100%
```

```{image} ../figures/real_des_patches/lead_jhelum_s_eastern_extension.png
:alt: Jhelum's eastern extension: the lead crosses the proposed track; a weak sequence
:width: 100%
```

```{image} ../figures/real_des_patches/lead_new_4__matched_filter_s_track.png
:alt: The matched filter's New-4 track: along the footprint's edge, a broad excess, no sequence
:width: 100%
```

```{image} ../figures/real_des_patches/lead_eastern_track_a.png
:alt: Eastern track A: along the footprint's edge, a broad faint excess, no distance peak
:width: 100%
```

```{image} ../figures/real_des_patches/lead_eastern_track_b.png
:alt: Eastern track B: along the footprint's edge, a broad faint excess, no distance peak
:width: 100%
```

```{image} ../figures/real_des_patches/lead_cetus_palca.png
:alt: The lead matched to Cetus-Palca: an excess at the faintest magnitudes only
:width: 100%
```

*Each lead: left, the map at its distance (dashed, the lead ±1°; green
dotted, the known track it was matched to, with the angle and overlap where
they meet); middle, the band's S/N against 200 null bands at each distance
(red circles: beyond all of them); right, the Hess difference with the
filter's polygon. `patches.py`, `lead_inspection`.*

| lead (found by) | its angle to the known track, overlap | band S/N, peak (distance) | Hess difference | verdict |
|---|---|---|---|---|
| ATLAS, control (both) | 3°, 24.5° | 9.4 (m−M 17) | a main sequence | the stream |
| **Leiptr** (line network) | **19°, 7.3°** | **5.2 (15.0, rising nearer)** | **a main sequence from g ≈ 20** | **a stellar stream along Leiptr's track** |
| "Jhelum's eastern extension" (line network) | 75°, 3.1° | 3.4 (15.5) | a weak, patchy sequence | crosses the proposed track; unconfirmed |
| "New-4" (matched filter) | 47°, 3.3° | 2.6 (16.0), flat | a broad excess, no sequence | the footprint's edge |
| eastern track A (line network) | 89° to New-4, 3.0° | 3.7 (16.0), flat | a broad excess at faint magnitudes | the footprint's edge |
| eastern track B (line network) | — | 3.5 (17.0), flat | a broad excess at faint magnitudes | the footprint's edge |
| "Cetus-Palca" (line network) | 20°, 9.0° | 3.5 (15.0), falling | an excess at g ≈ 24 only | not Cetus-Palca (26-35 kpc); depth |
| **NGC 1261's stream** (line network) | **10°, 11.2°** | 3.2 (15.5) | **a sequence from g ≈ 20.5** | **NGC 1261's stream, as Ibata et al. (2024) trace it** |
| **Tucana III, east of the DES 2018 track** (line network) | **16°, 3.3°** | **2.9 (17.0)** | **a main sequence from g ≈ 21** | **Tucana III's extension, at its distance** |
| Tucana III, west of the DES 2018 track (line network) | 22°, 0° | 3.2 (16.0) | a sequence from g ≈ 20 | a streak along the proposed extension, 2° north of the lead |

- **Leiptr** (Ibata et al. 2021) is the one lead that holds: the Hess
  difference shows a main sequence, the shape of a stellar population that
  no depth or galaxy artefact makes, along 7° of Leiptr's track. Its
  distance is uncertain. Leiptr is catalogued at 6-9 kpc (m−M 13.9-14.7),
  nearer than any distance we query, and the band's S/N is highest at the
  nearest (m−M 15); but the sequence's turnoff, at g ≈ 20, would place it at
  about m−M 15.8 by comparison with ATLAS (turnoff at g ≈ 20.8, m−M 16.8).
  Either Leiptr reaches farther here than catalogued, or this is a
  population beside it. The 19° between the two tracks, just past the 15° a
  match now needs, fits a curved stream fitted with a straight line in an
  11° window.
- **The eastern tracks** — the matched filter's "New-4", the line network's
  tracks A and B — run along the footprint's eastern edge, toward lower
  Galactic latitude. None peaks at a distance, and their Hess differences
  hold no sequence, only a broad excess at the faintest magnitudes at every
  colour: a change of depth or of galaxy contamination along the edge. They
  crossed New-4's track at 47-89°.
- **"Jhelum's eastern extension"** crosses the extension Ibata et al. (2024)
  propose at 75°. Its profile peaks at m−M 15.5, near Jhelum's 15.6, and its
  Hess difference shows a weak sequence — not enough to call it a stream.
- **"Cetus-Palca"** is at the wrong distance (Cetus-Palca lies at 26-35 kpc)
  and its excess lies at the faintest magnitudes only.
- **NGC 1261's stream** (Ibata et al. 2024): the lead runs along it (10°,
  over 11°), and the Hess difference shows a sequence whose turnoff, at
  g ≈ 20.5-21, puts it near m−M 16.5 — the globular cluster NGC 1261 is at
  m−M 16.1. The band's S/N peaks nearer (15.5), as Leiptr's does: the
  profile, through the matched filter's changing background with distance,
  prefers the near distances, and the sequence is the better guide.
- **Tucana III's extensions** (Ibata et al. 2024), found by the four-model
  line network beyond the DES 2018 track either side. East, the band's S/N
  peaks at m−M 17.0 — Tucana III's own distance — and the Hess difference
  shows a main sequence turning off at g ≈ 21: the stream continues, at its
  distance, 8° east of where DES 2018 stopped. West, the map shows a streak
  along the proposed extension itself, about 2° north of the lead, which
  crosses it at 22°; the Hess difference of the lead holds a sequence too —
  something is there, which the straight segment fits poorly.

```{image} ../figures/real_des_patches/lead_ngc_1261_s_stream.png
:alt: NGC 1261's stream: along the proposed track, a sequence in the Hess difference
:width: 100%
```

```{image} ../figures/real_des_patches/lead_tucana_iii__east_of_its_des_2018_track.png
:alt: Tucana III east of its DES 2018 track: a peak at its distance and a main sequence
:width: 100%
```

```{image} ../figures/real_des_patches/lead_tucana_iii__west_of_its_des_2018_track.png
:alt: Tucana III west of its DES 2018 track: a streak along the proposed extension, north of the lead
:width: 100%
```

## Reproducing

```bash
python scripts/experiments/real_des/patches.py line-sky --config "hough/band2 residual"   # the catalogue
python scripts/experiments/real_des/patches.py leads    # the leads' figures and leads.csv
```
