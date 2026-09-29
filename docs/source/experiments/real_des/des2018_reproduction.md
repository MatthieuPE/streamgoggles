# The DES 2018 analysis, reproduced

When a DES 2018 stream is missing from the network's results, is it missing
from the data, or did our cuts and matched filter lose it? To answer that
without the network, `notebooks/des2018_reproduction.ipynb` redoes the
analysis of Shipp et al. (2018, [arXiv:1801.03097](https://arxiv.org/abs/1801.03097))
on our DES Y6 Gold download, then measures every stream with the paper's
selection and with ours. The notebook runs end to end in about two minutes;
the figures below are its outputs.

## What follows the paper, and what cannot

| | DES 2018 | here |
|---|---|---|
| data | Y3A2 Gold, ngmix star/galaxy | Y6 Gold, `0 <= EXT_XGB <= 1` |
| sample | 16 < g < 23.5, 0 < g−r < 1 | the same |
| isochrone | Dotter et al. (2008), 13 Gyr, Z = 0.0002 | Marigo et al. (2017), 13 Gyr, Z = 0.0002 — Dotter 2008 is installed only at 10 and 12 Gyr (and ugali silently returns 12 when asked for 13); the two keep 85% of the same stars |
| filter | E = 2, C = (0.05, **0.10**), Δμ = 0.5, err(g) = 0.001 + exp((g − 27.09)/1.09) | the same, built by streamobs (whose "DES 2018" default padding is 0.05, not 0.10) |
| maps | HEALPix nside 512, Gaussian σ = 0.3° | the same |
| footprint | griz detection fraction > 50% | fraction of each pixel's nside-1024 sub-pixels holding a star > 50% |
| background | 5th-order 2D polynomial in the Galactic polar Lambert projection | `data_preparation.process_data_polyfit2d` and `polyfit2d` (degree 5 in each coordinate) |
| mask | Sagittarius, the LMC, the Galactic disk (extents not given); r < 1° around bright clusters and satellite and Local Group galaxies | Sagittarius to 9° from its track, the LMC to 20°, b > −35° towards the inner Galaxy; 1° around every catalogued cluster, dwarf and Local Group galaxy |

Outside-footprint and partly covered pixels never reach the fit: the fit mask
includes everything outside the footprint, the map handed to it holds no NaN,
and the smoothing is done on footprint pixels only (the function's own
smoothing fills masked pixels with the median first, which biases every edge).

## The paper's Figure 4

```{image} ../figures/des2018_reproduction/figure4.png
:alt: Residual density and polynomial background at m-M 16.7, with the DES 2018 streams drawn on the background
:width: 70%
```

*m−M = 16.7, Galactic polar Lambert equal-area, l = 120° up, on the paper's
colour scales. Top: residual density; bottom: the 5th-order background with
the streams drawn as the paper draws them (`objects_overlap.des2018_arc`).*

It reproduces: ATLAS, Phoenix, Turbio, Tucana III and the crowded quadrant of
Jhelum, Indus, Chenab and Ravi on a smooth background. The data differ in the
details — a fainter LMC periphery, more structure around Gold's foreground
holes — not in the streams.

## Our selection against DES 2018's

Each selection is a set of cuts and a filter; crossing them separates the two:

```{image} ../figures/des2018_reproduction/cmd.png
:alt: The two matched filters at m-M 16.7 over the colour-magnitude diagram, with the two sets of cuts
:width: 45%
```

*Both filters at m−M 16.7. They differ little in width; what differs is
depth: our sample runs a magnitude past g = 23.5.*

Each stream is then measured at its own distance (Table 1) along the paper's
track: the density profile across it, fitted with a quadratic beyond two
widths, and the stars within one width compared with it. Pixels near another
DES 2018 stream, masked or outside the survey are left out.

```{image} ../figures/des2018_reproduction/profiles.png
:alt: Density profile across each stream over its fitted background, DES 2018 selection and ours
:width: 100%
```

| stream | m−M | DES 2018 | DES 2018 cuts, our filter | our cuts, DES 2018 filter | ours | paper's significance |
|---|---|---|---|---|---|---|
| ATLAS | 16.8 | 21.8 | 24.5 | 20.5 | 23.1 | 13.9 |
| Elqui | 18.5 | 15.2 | 17.3 | 14.2 | 15.7 | 18.4 |
| Jhelum | 15.6 | 15.1 | 13.7 | 14.0 | 12.0 | 18.6 |
| Phoenix | 16.4 | 12.8 | 12.7 | 12.7 | 12.9 | 11.1 |
| Tucana III | 17.0 | 11.2 | 13.1 | 10.0 | 11.1 | 17.0 |
| Indus | 16.1 | 10.8 | 10.6 | 11.9 | 11.0 | 21.4 |
| Chenab | 18.0 | 10.4 | 10.9 | 11.5 | 11.7 | 15.1 |
| Turranburra | 17.2 | 8.6 | 10.1 | 3.0 | 3.9 | 14.4 |
| Aliqa Uma | 17.3 | 7.9 | 8.4 | 7.1 | 7.9 | 9.1 |
| Wambelong | 15.9 | 6.8 | 6.5 | 5.7 | 5.2 | 5.9 |
| Willka Yaku | 17.7 | 6.5 | 7.5 | 7.7 | 8.7 | 7.1 |
| Turbio | 16.1 | 3.9 | 3.9 | 5.5 | 5.1 | 7.9 |
| Molonglo | 16.8 | −0.7 | 0.1 | 0.2 | 1.1 | 5.2 |
| Ravi | 16.8 | −1.3 | −0.6 | 0.3 | −0.2 | 10.3 |

*S/N of each stream's core in each selection. The paper's significance
(Table 1) is a different, likelihood-based measurement; it ranks the streams.*

- **Twelve of the fourteen streams are in the data.** Ravi and Molonglo are
  not, even with the paper's own cuts, filter and track; both are measured on
  only half to 60% of their core, among other streams and masked galaxies,
  and are among the paper's weakest.
- **Our cuts and filter cost little**, except for Turranburra (8.6 → 3.9):
  the paper's cuts with our filter keep it, our cuts with the paper's filter
  lose it — our cuts, a magnitude deeper, add background there without adding
  the stream.

## Three tracks that are not the paper's

galstreams has no DES 2018 track for ATLAS, Aliqa Uma or Molonglo; the first
detection test used Li et al. (2021) for the first two and Grillmair (2017)
for Molonglo.

```{image} ../figures/des2018_reproduction/tracks.png
:alt: ATLAS, Aliqa Uma and Molonglo in the paper's frame, with the galstreams tracks dotted
:width: 100%
```

| stream | galstreams track | separation from the paper's, median / max | S/N along the paper's / galstreams' |
|---|---|---|---|
| ATLAS | Li et al. 2021 | 0.14° / 0.31° | 21.8 / 17.3 |
| Aliqa Uma | Li et al. 2021 | 0.60° / 1.45° | **7.9 / −0.1** |
| Molonglo | Grillmair 2017 | 2.00° / 4.68° | −0.7 / 0.9 |

Aliqa Uma is in the data; its first detection band missed it. Every
detection test now follows the paper's tracks.

## What it changed

The first training's results ({doc}`results`) had listed Ravi, Aliqa Uma and
Jhelum as absent from the network's input. Only Ravi (and Molonglo) are:
Aliqa Uma had the wrong track, and Jhelum lies across a steep density
gradient that side bands read as a deficit, where the profile fit sees S/N 12.
So the missed streams that are in the data are a problem of the network, not
of the data or of our cuts — which is what {doc}`recovery` then measures.
