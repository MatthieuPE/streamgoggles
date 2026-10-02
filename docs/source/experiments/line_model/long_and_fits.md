# Four long models, and the leads fitted

{doc}`line_followup` left two questions. One long line model per fold found
nine of the fourteen DES 2018 streams with the matched filter — the best
result, but one model per fold is a noisy measure: does it hold with four?
And the leads that held were straight segments at the search's coarse
distances: where do they lie, and how far?

Branch `line-model-long-fits`; code `scripts/experiments/line_model/run.py`
(`lead_fits`, and the configuration `hough/band2 residual long x4`).

**In short.** <!-- filled in below -->

## Four long models per fold

<!-- added when the six trainings end -->

## The leads fitted

Each lead that held, and ATLAS as the control, is fitted from its stars:

- **a distance**: the stars selected by the matched filter at each trial
  distance from m−M 13.5 to 18.5 in steps of 0.1 — below the search's grid,
  which starts at 15, for Leiptr — and the distance where the band along the
  lead (±0.4°) stands out most against flanking bands 1.5-3° either side,
  scaled by their valid areas;
- **a curved track**: at that distance, the stars in 1.5° bins along the
  lead, the excess across it fitted in each bin with a Gaussian over a
  linear background, and a quadratic through the bins' centres where the
  Gaussian stands at 2σ; and a width, the bins' median σ;
- **the distance again**, along the curved track, in a band of 1.5 widths.
  The curved track is kept only if it gathers more of the excess than the
  straight lead: with few bins fitted, a quadratic can swing off the stream.

```{image} ../figures/line_model/fit_atlas__control.png
:alt: ATLAS fitted: the curved track follows the stream; the distance peaks at m−M 16.8, the literature's 16.65
:width: 100%
```

```{image} ../figures/line_model/fit_tucana_iii__east_of_its_des_2018_track.png
:alt: Tucana III east of its DES 2018 track, fitted: the distance peaks at m−M 16.7, Tucana III's 17.0
:width: 100%
```

```{image} ../figures/line_model/fit_leiptr.png
:alt: The Leiptr lead fitted: the distance peaks at m−M 15.1, farther than Leiptr's catalogued 14.25
:width: 100%
```

```{image} ../figures/line_model/fit_ngc_1261_s_stream.png
:alt: NGC 1261's stream fitted: the distance peaks at m−M 15.5
:width: 100%
```

```{image} ../figures/line_model/fit_tucana_iii__west_of_its_des_2018_track.png
:alt: The lead west of Tucana III fitted: a wide structure on Indus's track
:width: 100%
```

*Left: the stars in the matched filter at the lead's best distance, along
and across the lead, each bin along it minus its own median; orange, the
fitted track and the bins it was fitted to (or the straight lead, where the
curve was rejected). Right: the band's excess S/N against flanking bands at
each trial distance, along the straight lead (blue) and the fitted track
(orange); green, the literature distance; grey, the search's distances.
`run.py`, `lead_fits`.*

| lead | track | distance (m−M) | S/N | literature | width | from the literature track |
|---|---|---|---|---|---|---|
| ATLAS, control | curved (15 bins) | **16.8** (16.5-16.9) | 21.7 | 16.65 (Li et al. 2021) | 0.22° (DES 2018: 0.24°) | 0.1° |
| **Tucana III, east of its DES 2018 track** | curved (5 bins) | **16.7** (16.5-17.4) | 7.4 | 17.0 (Tucana III) | 0.54° | 2.1° from Ibata et al.'s |
| **NGC 1261's stream** | straight | **15.5** (15.1-15.8) | 6.7 | 16.06 (the cluster) | — | 0.3° |
| **the Leiptr lead** | straight | **15.1** (14.9-15.4) | 12.7 | 14.25 (Leiptr, Ibata et al. 2021) | — | 1.1° |
| "Tucana III, west" | curved (4 bins) | 15.3 (15.2-15.6) | 15.5 | — | 1.0° | **on Indus's DES 2018 track** (0.3°) |

*The distance's range is where the S/N stays within 1 of its peak.*

1. **The fit recovers the control**: ATLAS at m−M 16.8 (16.65 catalogued),
   its width 0.22° (0.24°), its track within 0.1° of Li et al.'s; and the
   curved track raises its S/N from 15.6 to 21.7.
2. **Tucana III continues east of where DES 2018 stopped, at its
   distance**: m−M 16.7 (16.5-17.4), against Tucana III's 17.0, for 8°
   beyond the DES 2018 track's end, 2° from the path Ibata et al. (2024)
   propose there.
3. **NGC 1261's stream** lies on its proposed track (0.3°), at m−M 15.5
   (15.1-15.8), 0.5 mag nearer than its cluster (16.06) — a stream need not
   be at its progenitor's distance along all its length, and the matched
   filter's isochrone (13 Gyr, Z = 0.0002) is not NGC 1261's ([Fe/H] = −1.27).
4. **The Leiptr lead is not at Leiptr's distance**: m−M 15.1 (14.9-15.4),
   10.5 kpc, where Leiptr is catalogued at 6-9 kpc (m−M 14.25), and its
   main sequence turns off at g ≈ 20, as a population at about m−M 15.5-16
   would. Either Leiptr reaches farther than catalogued here, or this is
   another stream running near it (1.1° from its track, at 19°).
5. **The lead west of Tucana III was Indus**: its fitted track lies on
   Indus's DES 2018 track, and it is as wide (1.0° against 0.83°). The
   streak along Ibata et al.'s western extension of Tucana III, 2° north,
   is not what the fit followed.

## Reproducing

```bash
python scripts/experiments/line_model/run.py train --config "hough/band2 residual long" --seed 43   # 44, 45; --train-sky fold1; ~75 min each
python scripts/experiments/line_model/run.py evaluate --config "hough/band2 residual long x4"        # and --train-sky fold1; --sets fainter / "length scan"
python scripts/experiments/line_model/run.py line-sky --config "hough/band2 residual long x4"
python scripts/experiments/line_model/run.py fits    # the leads fitted
```
