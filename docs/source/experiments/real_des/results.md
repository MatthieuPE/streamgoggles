# First training: results

The maps the first training produced ({doc}`first_training`), and whether the
DES 2018 streams are found in them. Why the missed ones are missed is taken
further in {doc}`des2018_reproduction` and {doc}`recovery`.

## The maps

```{image} ../figures/real_des/prediction.gif
:alt: The network's output over the DES footprint, stepping through the queried distance moduli from 15 to 19
:width: 100%
```

*The out-of-fold ensemble output over the inference sky, one frame per
queried distance modulus from 15 to 19; the DES 2018 tracks are outlined in
orange. Sagittarius, the clusters and the dwarfs are masked (white).*

```{image} ../figures/real_des/prediction_dm16.5.png
:alt: Network output at m-M 16.5
:width: 100%
```

*m−M 16.5 (20 kpc). ATLAS is the dark line on its track near RA 15-30°,
Dec −25° to −33°. The strips along the edge of the Sagittarius mask and at the
footprint's edges are real structure the masks left, taken up below.*

```{image} ../figures/real_des/prediction_dm18.5.png
:alt: Network output at m-M 18.5
:width: 100%
```

*m−M 18.5 (50 kpc): the Magellanic Clouds' distance, and the southern edge of
the footprint lights up toward them. Elqui and Chenab are at their own
distances here; rings surround the masks of the Sculptor and Fornax dwarfs.*

### How much the folds mattered

The ensembles also predicted their own training stripes, which is what a
model trained on the whole sky would have done. On stream-free sky, the
fraction of pixels above 0.5:

| m−M | from the ensemble that never saw the pixel | from the one that trained on it | ratio |
|---|---|---|---|
| 15.0 | 0.42% | 0.15% | **2.8×** |
| 15.5 | 0.52% | 0.18% | **2.9×** |
| 16.0 | 1.13% | 0.69% | 1.6× |
| 16.5 | 2.16% | 1.78% | 1.2× |
| 17.0 | 3.18% | 2.83% | 1.1× |
| 19.0 | 3.21% | 2.95% | 1.1× |

A model scored on its own training sky would have shown a false-alarm rate
**nearly three times too low at m−M 15-15.5**, and about 10% too low beyond
m−M 17. The folds were worth their cost.

### What the stream-free sky still holds

About 3% of the training mask's sky was above 0.5 beyond m−M 17 — far more
than any simulated sky gave. The maps show why: it is not noise but real
structure the masks left. The fraction of flagged pixels around each
structure, at its own distance, against distance from it:

| structure | flagged pixels by distance | radius in the training mask | calibration radius |
|---|---|---|---|
| LMC | 67% at 9-12°, 52% at 12-15°, 9% at 15-18°, 3% (background) at 18-21° | not masked | **20°** |
| SMC | 90% at 6-9°, 23% at 9-12°, 1% at 12-15° | not masked | **12°** |
| Sagittarius | 45% at 6-7° from its track, 8% at 8-9°, background at 9-10° | 6° | **9°** |
| Sculptor, Fornax | 37% and 18% at 1-2°, nothing beyond | 0.94°, 1.5° (5 half-light radii) | **12 half-light radii** |
| NGC 1904, NGC 1261 | 72% and 86% within 1°, about 4% at 1-2° | not selected: their centres sit in Gold's foreground holes | **2°**, for any cluster within 2° of the footprint |

The Magellanic Clouds light up at exactly their distance, and the dwarfs'
stars extend past their masks — the network is finding real stellar
structure. For measuring false alarms, these are removed: the **calibration
sky** is the training mask's sky minus these regions, 3,331 of its 4,017 deg².
On it, the fraction above 0.5 falls to 0.2-0.8% at every distance, while the
removed 17% of sky held up to 18% flagged pixels: **80-85% of the apparent
false alarms beyond m−M 16.5 were these structures**. The training itself
still saw them, taught as background; whether masking them in training too
changes the models is left for a later run.

## The DES 2018 streams

### The test

The criterion of the simulated experiments, adapted to real tracks
(`evaluation.footprint`: `false_alarm_map`, `track_band`,
`real_track_statistics`):

- **A false-alarm-rate map per distance.** Each pixel gets the fraction of
  calibration pixels *of its own fold* that the network scored at least as
  high. The two folds' ensembles are calibrated separately, so one cut then
  means the same thing everywhere; ties count against the pixel.
- **Flagged**: a false-alarm rate of 10⁻³ or less.
- **The band**: pixels within one width of the stream's track **as the paper
  draws it** (`objects_overlap.des2018_arc`: great-circle arcs between the
  Table 1 end points; ATLAS its polynomial). A first run took the tracks from
  `galstreams` (Shipp et al. 2018, 2019 where available, Li et al. 2021 for
  ATLAS and Aliqa Uma, Grillmair 2017 for Molonglo); the DES 2018
  reproduction showed Aliqa Uma's and Molonglo's to run 0.6-2° off the
  streams, and the test was re-run along the paper's. It changed no verdict.
- **The null bands**: the band's pixels moved rigidly onto 200 random places
  and orientations of the calibration sky, kept when at least 90% land on it.
- **Detected**: at least 20 flagged pixels in the band, and a flagged density
  standing out from the null bands at S/N ≥ 2 (`band_snr`), at the queried
  distance nearest the stream's own.

### The result: 5 of 14

| stream | m−M | queried | flagged / band pixels | S/N | |
|---|---|---|---|---|---|
| Elqui | 18.5 | 18.5 | 280 / 829 | 108 | ✔ |
| ATLAS | 16.8 | 17.0 | 148 / 846 | 31 | ✔ |
| Willka Yaku | 17.7 | 17.5 | 64 / 215 | 22 | ✔ |
| Tucana III | 17.0 | 17.0 | 41 / 117 | 18 | ✔ |
| Chenab | 18.0 | 18.0 | 62 / 2033 | 7.2 | ✔ |
| Molonglo, Wambelong, Turbio, Aliqa Uma, Turranburra, Phoenix, Indus, Ravi, Jhelum | 15.6-17.3 | | 0 flagged | < 0 | ✘ |

```{image} ../figures/real_des/snr_by_distance.png
:alt: S/N along each stream's track at every queried distance
:width: 100%
```

*S/N along each stream's track at every queried distance; dashed, its
catalogued distance. The five detections peak at or next to it — Elqui at
18.5 exactly, ATLAS at 16.5 for 16.8, Chenab at 18.5 for 18.0 — which is what
ties each detection to its stream. Turbio and Turranburra light up only at
18.5-19, far from their own distances: Turbio's southern end runs within about
13° of the SMC, and neither excess is the stream.*

### Why nine are missed

Two measurements separate a stream the network misses from one the input
does not contain: the classic matched-filter significance of the stream in the
counts the network reads (the band against side bands two to four widths
away), and the network's own S/N.

```{image} ../figures/real_des/input_vs_network.png
:alt: The network's S/N against each stream's S/N in the matched-filter counts, coloured by distance
:width: 90%
```

*Each stream's S/N in the input counts (x) against the network's (y), coloured
by its distance, along the paper's tracks. Every detected stream is at m−M
16.8 or beyond (dark); the strongly visible missed ones, Phoenix and Indus, are
nearer than 16.5 (pale). Aliqa Uma (5.5, m−M 17.3) is the exception. Jhelum's
input S/N is the side-band bias discussed below.*

They fall into three groups:

- **Not in the input at all**: Ravi (input S/N −1.0), Aliqa Uma (−1.8) and
  Jhelum (−2.8), as first measured, along the galstreams tracks. Along their DES tracks, with this filter and these cuts, the
  matched-filter counts show no excess; no model reading these maps could find
  them. **Corrected since** (see the next section): this holds for Ravi only.
- **An excess, but not at one distance**: Turbio's excess is 6-9% at every
  channel distance from 14.5 to 19.5, and Molonglo's 8-11% from 14.5 to 17.5.
  A stream is concentrated at its distance; a flat excess is a density
  difference across the band, and the network declining to call it a stream
  is arguably right. Wambelong and Turranburra show marginal inputs (S/N 3.8
  and 4.9).
- **In the input, and missed**: Phoenix (input S/N 11.7, contrast 22%) and
  Indus (12.4, 6%). Phoenix is the clearest case: its excess peaks at its own
  distance and falls off on both sides, the same profile as ATLAS, and the
  counts show it as a thin line — yet the network responds only in patches,
  reaching false-alarm rates of about 10⁻², not 10⁻³. It is not the
  per-pixel statistic: averaging the output over the whole band gives Phoenix
  S/N 1.1 and Indus 0.3, against 11-28 for the detected streams.

**The pattern is distance**: everything detected lies at m−M ≥ 16.8,
everything visible but missed at m−M ≤ 16.4. It is the weakness the
simulations measured ({doc}`../stream_parameters`, conclusion 3: at fixed surface
brightness, closer streams are harder, and m−M 15-16 is the hard regime), now
seen on real streams.

### Corrected by the DES 2018 reproduction ({doc}`des2018_reproduction`)

`notebooks/des2018_reproduction.ipynb` redoes the paper's own analysis on our
data: its cuts (16 < g < 23.5, 0 < g−r < 1), its filter (13 Gyr, Z = 0.0002,
E = 2, C = (0.05, 0.10), its error model), nside 512 maps smoothed at 0.3°,
a 5th-order polynomial background in the Galactic polar Lambert projection,
and its Fig. 4. Then it measures every stream at its own distance, along the
paper's Table 1 track (`objects_overlap.des2018_arc`), with the paper's
selection, ours, and the two crossings of them. Three of the statements
above do not survive:

- **Aliqa Uma is in the input** (S/N 7.9 with our selection). Its detection
  band followed the galstreams track of Li et al. (2021), which runs 0.6° from
  the paper's on average and 1.45° at worst — more than five widths. Along
  that track the same measurement gives −0.1.
- **Molonglo's track is not DES's**: Grillmair (2017), 2° from the paper's.
  Along the paper's, Molonglo shows nothing either (S/N 1.1), so it moves to
  the "not in the data" group with Ravi, rather than "an excess at every
  distance".
- **Jhelum is in the input** (S/N 12 with our selection). The side-band
  statistic above averages a steep density gradient across Jhelum, and a
  diffuse overdensity on one side, into its background, and reads the stream
  as a deficit. The notebook fits a quadratic to the density profile across
  the stream beyond two widths instead.

**Ravi alone is absent** — with the paper's own cuts, filter and track too, so
from the data (Y6 Gold, EXT_XGB stars), not from our choices. Our cuts and
filter cost little elsewhere: the same S/N as the paper's selection within a
few units, except **Turranburra (8.6 → 3.9)**, which the crossings trace to
our cuts — a magnitude deeper — not to our filter.

So among the missed streams, only Ravi and Molonglo are missing from the
input; Jhelum, Phoenix, Indus and Turbio are all in it, and all at
m−M ≤ 16.4. Re-run along the paper's tracks, the detection test gives the
same five detections: **Aliqa Uma, now in the input** (side-band S/N 5.5 at
m−M 17.5), **has no flagged pixel at any distance** — a stream at 17.3, the
network's good range, that it does not see. Jhelum's side-band input S/N
along the paper's track is −6.9, the bias described above; the notebook's
profile fit gives 12.

