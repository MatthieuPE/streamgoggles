# Following up the line model: more models, and the leads

{doc}`line_sky` searched the real DES sky with the line model and left two
questions: does the 2° line model gain from more models or longer
training — enough to find Jhelum and Phoenix again — and are the tracks it
found outside the DES 2018 streams real? This page answers both.

Branch `line-model-followup`; code `scripts/experiments/real_des/patches.py`
(`lead_inspection`, and the configurations `hough/band2 residual x4`,
`s42` and `long`).

**In short.** <!-- filled in below -->

## More models, longer training

<!-- the comparison is added below -->

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

## Reproducing

```bash
python scripts/experiments/real_des/patches.py line-sky --config "hough/band2 residual"   # the catalogue
python scripts/experiments/real_des/patches.py leads    # the leads' figures and leads.csv
```
