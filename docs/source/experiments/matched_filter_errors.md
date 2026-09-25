# Matched filter errors on DES

The matched filter decides which stars reach the model, so its width decides
both what the model sees and what its label contains. Its width comes from a
photometric error model — and until this experiment, the one in use was not
DES's.

**Decision: use the DES Y6 error model, `DES_YR6_ERROR_MODEL`, at one sigma**
(`"error_model": DES_YR6_ERROR_MODEL, "error_multiplier": [1.0, 1.0]` on the
isochrone filter). It keeps 77-93% of a stream's own stars from m−M 15 to 19,
against 54-77% for the filter in use, at a signal-to-noise between 4% below
and 5% above it.

## The filter, and the two things that set its width

The filter (streamobs's `build_match_filter`) is the isochrone — 13 Gyr,
Z = 0.0002, Marigo2017 — shifted to the trial distance, with each colour edge
pushed out by `error_multiplier` times the photometric error at that
magnitude, plus a distance spread of ±0.5 mag and a small colour padding. It
stops at the main-sequence turnoff (`rgb_clip_mag`).

Two separate things therefore set its width, and this page treats them
separately:

- **the error model**, σ(g): how far stars actually scatter around the
  isochrone at each magnitude. That is a property of the survey, to be
  measured, not chosen;
- **the error multiplier**: how many σ the filter spans on each side. That is
  a choice, a trade between keeping the stream's stars and admitting
  background.

## What the pipeline had: LSST errors

`build_match_filter` called without an error model — as every filter so far
was — uses streamobs's `default_errors`, fitted on **LSST DC2**. DES's errors
are 4.7 to 7 times larger over the magnitudes the filter covers:

| g | DES Y6 (adopted fit) | LSST DC2 (streamobs default) | ratio |
|---|---|---|---|
| 20 | 0.024 | 0.0050 | 4.7× |
| 22 | 0.035 | 0.0064 | 5.4× |
| 23 | 0.057 | 0.0092 | 6.1× |
| 24 | 0.114 | 0.0169 | 6.8× |
| 24.5 | 0.172 | 0.0247 | 7.0× |

So the filter drawn around the LSST errors is far narrower than the scatter of
a DES stream's stars: on DES it keeps **76% of a stream's stars at m−M 15,
72% at 17, and 54% at 19**.

## Fitting the DES Y6 errors

The errors come from streamobs's own DES Y6 model, so the filter and the
injected streams describe the same survey:

- **which errors**: `get_photo_error(kind="sample")`, the scatter of observed
  around true magnitudes measured on the Y6 Balrog injections — what actually
  moves a star off the isochrone. The *reported* errors (`kind="catalog"`) run
  about 1.46 times smaller and would give too narrow a filter;
- **at which depth**: the 16th percentile of each band's magnitude-limit map,
  taking the shallower band (r, 24.72), so the filter is wide enough for the
  shallower parts of the footprint;
- **over which range**: g = 16 to 24.5, the faint end of the analysis clip —
  as faint as the filter ever selects.

`fit_survey_error_model()` does this and returns

| parameter | value |
|---|---|
| `baseline_error` | 0.0216 |
| `exp_pivot` | 26.457 |
| `exp_scale` | 1.032 |

for σ(g) = baseline + exp((g − pivot) / scale). The baseline is large because
the Balrog scatter itself has a 0.02 mag floor at the bright end, where the
reported errors go down to 0.005.

```{image} figures/matched_filter_errors/error_fit.png
:alt: DES Y6 photometric scatter, the adopted fit, a fit over a wider range, and the LSST default
:width: 100%
```

*Top: the DES Y6 scatter the model is fitted to (points), the adopted fit
(solid), a fit over a wider range (dashed, not used), the LSST DC2 default the
filter used until now (blue), the reported DES errors, and the DES 2018
constants streamobs also carries. Bottom: each fit relative to the scatter.
The shaded region is past the clip, where the filter never selects a star.*

**Why the fit stops at 24.5.** A first fit ran to two magnitudes past the
depth, 26.7. Beyond about 24.7 the Balrog scatter flattens and then dips — the
behaviour of objects near the detection limit — which a rising exponential
cannot follow, and chasing it pulls the model **24% below the true scatter at
g = 23.5-24.5**, exactly where a distant stream's stars are:

| fit range | worst residual, g 16-24.5 | rms | bias at g 23.5-24.5 |
|---|---|---|---|
| 16 to 26.7 | 33% | 11% | −24% |
| **16 to 24.5 (adopted)** | **13%** | **5%** | **−1.5%** |

Pivot and scale are strongly correlated in this model, so fits over different
ranges can reach quite different parameters and still describe the same
magnitudes well; it is the curve that is compared, never the parameters one by
one. A test checks that `DES_YR6_ERROR_MODEL` still matches its fit.

## The filters

Four filters around the same isochrone: the one in use, and the DES errors
at one, one and a half, and two sigma — streamobs's default multiplier is two.

```{image} figures/matched_filter_errors/filters.png
:alt: The four filters in colour-magnitude space over a simulated DES stream
:width: 100%
```

*Each filter over the observed stars of a simulated DES stream at that
distance (grey), produced by the training pipeline and cut to the 16-24.5
clip; the legend gives the fraction of those stars each filter keeps. The
LSST-error filter (blue) is visibly narrower than the stream's scatter, most
at the faint end. Every filter stops at the turnoff; the giant branch and the
horizontal branch above it are 1%, 3% and 10% of a stream's visible stars at
m−M 15, 17 and 19.*

## Completeness against signal-to-noise

A wider filter keeps more of the stream but also admits more background, so
keeping more stars is only useful if it does not cost more in noise. The
figure of merit is the one for counting a uniform overdensity: stream stars
kept over the square root of background stars kept, S/√B. Stream stars come
from three simulated streams per distance (thousands of stars, so the
fractions are good to about a point); background stars are the whole masked
DES Y6 background, 31.6 million stars. S/√B is shown relative to the filter
in use, which makes it independent of the stream's surface brightness and of
the area it covers.

```{image} figures/matched_filter_errors/completeness_snr.png
:alt: Fraction of the stream kept and relative S/sqrt(B) against distance modulus, for the four filters
:width: 100%
```

*Left: the fraction of a simulated stream's stars each filter keeps. Right:
S/√B relative to the filter in use (dotted line).*

| m−M | in use (LSST, 2σ) | DES 1σ | DES 1.5σ | DES 2σ |
|---|---|---|---|---|
| 15 | 76% | 91% (0.98×) | 95% (0.92×) | 97% (0.87×) |
| 16 | 76% | 92% (0.98×) | 96% (0.92×) | 98% (0.86×) |
| 17 | 72% | 90% (0.96×) | 94% (0.89×) | 96% (0.83×) |
| 18 | 63% | 84% (1.00×) | 90% (0.93×) | 93% (0.86×) |
| 19 | 54% | 77% (1.05×) | 84% (0.99×) | 88% (0.93×) |

*Fraction of the stream kept, and in brackets S/√B relative to the filter in
use.*

The DES errors at one sigma keep 15 to 23 more points of the stream than the
filter in use, at an S/√B between 0.96 and 1.05 times its own: slightly below
it from m−M 15 to 17.5, level at 18, above it beyond. Each further half sigma
adds a few points of completeness and costs 5-7% of S/√B.

All three DES filters dip at m−M 17. It is a matter of timing: between 16.5
and 17 the background they admit grows fastest relative to the filter in use
(from 1.59 to 1.69 times as much, at one sigma), while the extra stream stars
they keep barely change; beyond 17 the filter in use starts losing stream
stars quickly, and the DES filters pull ahead.

### Does it hold for other streams and other skies?

The same measurement was repeated two ways, with the same ranking every time:

- **Streams whose population is not the filter's.** Six populations drawn
  as the adopted training set draws them (age 9-13.5 Gyr, Z 0.0001-0.001).
  DES 1σ keeps 74-91% of them and is at 0.98-1.08× the filter in use; 1.5σ
  0.92-1.04×, 2σ 0.86-0.98×. A narrower filter might have been expected to
  suffer most from a mismatched isochrone; at one sigma of the correct errors
  it does not.
- **Dense and sparse sky.** The background split at the median galactic
  latitude (47°): one sigma is 0.96-1.05× in the denser half and 0.96-1.04×
  in the sparser one.

## Conclusion

1. **The filter in use had LSST errors, 4.7 to 7 times smaller than DES's**,
   and on DES it keeps only 54-77% of a stream's own stars.
2. **The DES Y6 error model is `DES_YR6_ERROR_MODEL`**, the Balrog scatter fitted
   over g = 16-24.5, where it follows the scatter to 5% rms. Fitting past the
   clip biased it by −24% at the faint end and was not used.
3. **Use it at one sigma.** It keeps 77-93% of the stream from m−M 15 to 19,
   at a signal-to-noise 0.96 to 1.05 times that of the filter in use, for
   streams of the filter's population and of others, in dense and in sparse
   sky. One and a half and two sigma buy a few more points of completeness
   and cost up to 11% and 17% of the signal-to-noise.

In a filters configuration:

```python
from streamgoggles.matched_filter import DES_YR6_ERROR_MODEL

"good": {
    "type": "isochrone",
    "reference_isochrone": {"age": 13.0, "z": 0.0002},
    "error_model": DES_YR6_ERROR_MODEL,
    "error_multiplier": [1.0, 1.0],
}
```

The completeness matters beyond the figure of merit: the label is "the stream
stars the filter selects", so a filter that keeps half of a distant stream
gives the model half a label.

### Caveats

- **S/√B is a counting proxy.** The model reads the maps, not a count, and
  may use the extra stars differently; the definitive comparison would train
  on each filter and compare detection, which has not been done.
- **One depth for the whole footprint.** The error model is fitted at the 16th
  percentile of the depth; where the survey is deeper, the filter is a little
  wider than one sigma of the local errors.
- **One band.** streamobs widens the colour edges with a single error curve in
  the magnitude band, fitted here in g; the colour's own error, which also
  carries r's, is folded into that curve and the multiplier.
- **The stream-parameters experiment keeps its filter.** Its trained models
  depend on it; the calibrated filter is for the real-data training.

## Reproducing

From the repository root, in the `streamml` environment, with the real
background built (see {doc}`../narrative/real_des_background`):

```bash
python scripts/experiments/matched_filter_errors/run.py
```

It fits the error model, simulates the streams, measures every filter on the
background, writes the figures to
`docs/source/experiments/figures/matched_filter_errors/` and the numbers to
`data/experiments/matched_filter_errors/results.csv`, and prints the tables
above. About ten minutes. The notebook `notebooks/define_matchedfilter_params.ipynb`
walks through the same comparison interactively.
