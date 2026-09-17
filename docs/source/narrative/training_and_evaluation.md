# Training and evaluation

```{note}
This page assumes familiarity with epochs/batches/optimizer steps and the
metrics it names (Dice, IoU, precision/recall, correlation) — see
{doc}`ml_concepts` first if any of those need a primer.
```

## `PlainTrainer` ({py:mod}`streamgoggles.training.plain_runner`)

A minimal, explicit, single-device training loop — no framework magic,
every step readable end to end. It consumes the plain dicts
{py:class}`~streamgoggles.datasets.stream_map_dataset.StreamMapDataset`
produces (`map_stack`/`label_stack`/`valid_mask`), not a bare tuple.

- `train_epoch(dl)` / `validate(dl, metrics_fn=None)` — one pass over a
  `DataLoader`, batch-size-weighted loss (and, for `validate`, any
  `metrics_fn` results too, so uneven final batches don't skew the
  average).
- `train(...)` — the full loop: epochs, optional validation, optional
  checkpointing via a {py:class}`~streamgoggles.storage.ModelStore` (pass a
  `config` dict containing a `"model_config"` key — the exact kwargs to
  reconstruct the model class — and it doubles as the store's addressing
  key), optional per-epoch callbacks. The final epoch is always
  checkpointed regardless of `save_every`.
- **Mixed precision on CPU.** The reference sketch this was built from
  assumed a CUDA-only `torch.cuda.amp.GradScaler()`; this dev environment
  is CPU-only, so `PlainTrainer` instead uses the device-aware
  `torch.autocast(device_type=..., enabled=amp)` +
  `torch.amp.GradScaler(device_type, enabled=amp and device_type=="cuda")`
  — autocast still applies (bfloat16) on CPU when `amp=True`, while the
  scaler (only meaningful for CUDA fp16) stays a no-op there instead of
  erroring.

`training/hyrax_runner.py` — a thin `hyrax`-orchestrated alternative to
`PlainTrainer` for distributed/production training — remains an explicit
stub (`NotImplementedError`), intentionally deferred until after a plain
training run was validated. `UNet` and `StreamMapDataset` themselves stay
completely free of `hyrax` imports either way, so the plain path never pays
for orchestration machinery it isn't using.

Its eventual shape is already decided, from reading `hyrax`'s own
registration mechanism directly rather than guessing: `hyrax.models.
model_registry.hyrax_model` is a class decorator for a `torch.nn.Module`
subclass needing `__init__(self, config, ...)`, `train_batch`,
`infer_batch`, and (recommended) a `prepare_inputs` staticmethod;
`hyrax.datasets.HyraxDataset` is a base class needing `__init__(self,
config)` and `__len__`, auto-registered via `__init_subclass__`. The plan
is two small adapter classes living in this one file — never new modules,
per an explicit "avoid file proliferation" steer — a `@hyrax_model`-
decorated `UNet` subclass that delegates `train_batch`/`infer_batch` to
the plain `UNet.forward` plus a `models.losses` loss, and a `HyraxDataset`
subclass that wraps (composes, doesn't reimplement) a plain
`StreamMapDataset` and delegates `__len__`/`__getitem__` to it. The
standard hyrax adapter-per-side pattern, at the narrowest footprint that
pattern allows.

## Evaluation ({py:mod}`streamgoggles.evaluation`)

### Metrics ({py:mod}`streamgoggles.evaluation.metrics`)

`iou`, `dice`, `precision_recall`, `confusion_matrix`, `mse`,
`correlation`, `weighted_recall`
— all pure-numpy, all `valid_mask`-aware (true exclusion, same convention
as the losses), all operating on whatever scale `pred`/`target` are
actually in: a `[0, 1]` probability for the binary/density label options
and for the current default `"stream_detection"` (whose label is already
`{0, 1}`, so `threshold=0.5` is a natural cut), or a literal star count for
`"stream_count"` (where `threshold` should be chosen accordingly, e.g.
`0.5` stars for "is there a true member here").

`confusion_matrix` returns the raw `tp`/`fp`/`tn`/`fn` counts every other
threshold-based metric here is derived from. Worth having separately
because the ratios hide *how* a model is wrong — 5 false positives and
5000 can give identical recall — and because none of the others surfaces
true negatives at all. One deliberate choice inside it: invalid pixels are
excluded from `tn` rather than counted as correct rejections, which would
otherwise inflate it by the entire out-of-footprint area and make any
TN-based rate (specificity, accuracy) meaningless.

Every metric with an undefined ratio — `iou`/`dice` with nothing predicted
and nothing true, `precision` with no positive predictions, `correlation`
with fewer than two valid pixels or zero variance — returns `float("nan")`
rather than a guessed `0.0`/`1.0`, so it can never silently read as a real
score. Aggregating code is expected to `.dropna()` explicitly.

`weighted_recall` is worth calling out: it weights recall by the target's
own magnitude, so missing a pixel with many true stream stars costs more
than missing a nearly-empty one — a natural fit for the literal-count
`"stream_count"` label; against `"stream_detection"`'s binary target it
reduces to plain recall (every true pixel weighs the same, `1`), which is
still a well-defined, useful number, just not doing anything extra beyond
ordinary recall in that case.

### Baseline ({py:mod}`streamgoggles.evaluation.baseline_threshold`)

`build_baseline` — a trivial, non-learned *k·σ* threshold on a finalized
map (mean + *k* standard deviations over valid pixels). Exists purely as a
sanity floor: if the network can't beat this, it hasn't learned anything a
simple threshold couldn't already do.

### Per-parameter recovery ({py:mod}`streamgoggles.evaluation.completeness_purity`)

`evaluate_on_grid(model, eval_dataset, metrics, ...)` walks every point of
an eval-mode `StreamMapDataset`, runs the model, computes the requested
metrics, and joins the result with that point's parameters into a
`pandas.DataFrame` — one row per sample, `id` set to the eval-grid index
(a stable identity). `plot_recovery_vs_parameter` plots any metric column
against any parameter column, optionally overlaying a baseline column.
Most metric names map to one column; `"precision"`/`"recall"` share a
single call, and `"confusion"` expands into four (`tp`/`fp`/`tn`/`fn`).

`aggregate_over_replicates(results, metrics, group_by=None)` is the
companion for grids built with
`build_eval_grid(..., n_replicates=k)` (see {doc}`datasets_and_models`):
it collapses the k rows per parameter combination into
`<metric>_mean`/`<metric>_std`/`<metric>_n`. Use it whenever a
per-parameter number is going to be quoted or plotted — a single
realization's Dice can sit anywhere inside that spread, so the mean±std
is the honest form of the same claim. Pass `group_by` explicitly when the
results carry non-parameter columns you don't want treated as grouping
keys (raw confusion counts, for instance); the default treats every
column that isn't a requested metric, `id`, or `replicate` as a parameter.

`compute_completeness_purity(results, stream_detection_fn, metric, ...)`
turns a metric column and a detection rule into completeness/purity
numbers. One real gap in the original design here: `results` from
`evaluate_on_grid()` on a `StreamMapDataset` eval grid contains **only true
streams** (eval mode always injects a real stream, unlike training mode's
`background_fraction`), so "purity" as classically defined (fraction of
detections that are true streams) is trivially 1.0 with nothing to
contaminate it. This function treats every row as a true stream unless
`results` carries an explicit `is_true_stream` boolean column — a real,
non-trivial purity needs that column populated with some background/
non-stream rows, e.g. by concatenating a stream eval grid's
`evaluate_on_grid()` output with a separately-evaluated background-only
set.

### Footprint-level detection ({py:mod}`streamgoggles.evaluation.footprint`)

Everything above scores one window at a time. The end product, though, is
a **HEALPix map** of where streams are, so the model is also scored on the
map itself.

**From windows to a HEALPix map.** `tiles_around_stream` lays overlapping
tiles (stride = half a window by default) over the part of the footprint
around an injected stream. `predict_footprint` runs the model on each tile
and projects every tile back to HEALPix with
{py:func}`~streamgoggles.matched_filter.stitch_windows_to_healpix`. Where
tiles overlap, a pixel takes its value from the tile where it lies
**furthest from an edge**. Overlapping outputs are **not averaged**: the
output is a probability that gets thresholded, and the mean of a confident
0.9 and a confident 0.1 is not a real 0.5. This is the U-Net
overlap-tile strategy. The input and the detection label are stitched the
same way, so all three maps line up pixel for pixel.

**The four fractions.** `score_footprint` counts `tp`/`fn`/`fp`/`tn` over
the stitched map and normalizes each row of the confusion matrix
({py:func}`~streamgoggles.evaluation.metrics.confusion_rates`):

| | predicted stream | predicted no stream |
|---|---|---|
| **true stream** | `tpr` = found | `fnr` = missed |
| **true no stream** | `fpr` = false alarm | `tnr` = correct reject |

Each row sums to one, so the numbers can be compared between maps whose
stream and background areas differ by orders of magnitude. Raw counts
cannot. Two details matter:

- Only pixels that some tile covered **and** that are finite in both maps
  are scored. A stitched map is NaN in footprint holes and outside the
  tiles, and `NaN > 0.5` is `False`. Without that mask, all the unobserved
  sky would count as correctly rejected background.
- If a stream is too faint to leave any pixel above `count_threshold`,
  `tpr`/`fnr` are **NaN, not 0**. "Nothing to detect" (the *label* has
  vanished) and "detected nothing" (the *model* missed it) mean opposite
  things, so they are kept apart.

The false-alarm rate is measured **near the stream** (inside the tiled
region), not over the whole survey.

**Averaging over realizations.** `evaluate_footprint_realizations` injects
`n_realizations` independent full-sky realizations for each parameter set
(seeded by `[seed, set_index, realization]`, so the run is reproducible and
every sky is independent). It returns one row per realization.
`aggregate_over_replicates(..., group_by=["richness"])` then gives
mean/std/n per surface brightness. A realization with no stream pixels is
kept as a row with NaN rates rather than dropped, so failures at the faint
end don't disappear. `n_true_pixels_n` against `tpr_n` counts how many
realizations still had a label. `score_footprint` also returns
`precision` (`tp / (tp + fp)`), the fraction of flagged pixels that are
really stream.

**The no-stream control.** A found fraction means nothing without the
background it has to stand out from: finding half the stream pixels is
worthless if half the background is flagged too. So by default
(`no_stream_control=True`) every realization is also predicted on
`background_only_sky`: the same sky with the stream removed, scored on
**the same tiles**. That gives `fp_no_stream`/`tn_no_stream`/
`fpr_no_stream`, the false-alarm rate set by the background alone. The gap
between `fpr` and `fpr_no_stream` is what the stream itself adds.

**Plots.** `plot_confusion_matrix` draws the averaged 2×2 matrix.
`plot_detection_rates` draws two panels against a parameter. The top panel
shows the found fraction (missed is its complement, so it isn't drawn). The
bottom panel shows the background false-alarm rate with the stream and
without it. Std bars are clipped to [0, 1], and points averaged over only
some realizations are annotated `k/n`.

**Any threshold, after the fact.** Passing `thresholds=THRESHOLD_GRID` to
`evaluate_footprint_realizations` also records, for each realization, how
many stream pixels (`n_above_stream`), background pixels
(`n_above_background`) and no-stream-control pixels (`n_above_no_stream`) lie
above each of 241 thresholds (evenly spaced in logit, so the tail near 1 is
resolved; 0.5 is included exactly). No threshold has to be chosen before
scoring.

`detection_metrics(results, THRESHOLD_GRID, group_by=["richness"], at=...)`
turns those counts into the area-independent metrics used to compare models
({doc}`../experiments/index`). With counts summed over the realizations of a
group, at threshold $t$: $S_t$ true stream pixels, $S_s$ of them above $t$;
$B_t$ true background pixels, $B_s$ of them above $t$;

$$
C = \frac{S_s}{S_t}\ \text{(completeness)},\qquad
F = \frac{B_s}{B_t}\ \text{(contamination)},\qquad
\frac{C}{F}\ \text{(contrast)}.
$$

$F$ counts at least one flagged pixel, so it is never zero and the contrast is
never overstated; the contrast is NaN (undefined, not zero) when no stream
pixel is found. Unlike purity, $S_s/(S_s+B_s)$, none of the three depends on
how much background was scored. `plot_detection_metrics` draws the three
against a parameter, one line per threshold, with points based on fewer than
20 found stream pixels drawn hollow.

**Per stream: is it detected?** Pixel completeness can be low while a stream
is still clearly visible as a line of flagged pixels. With both `thresholds`
and the no-stream control, `evaluate_footprint_realizations` also calls
`track_band_statistics` for each realization (for streams with a `width` and
`length`). The injector records where the stream was placed
(`inject_stream_full_sky(...)["placement"]`), and `stream_frame_coordinates`
turns any sky position into the stream's own $(\phi_1, \phi_2)$. It then counts
flagged pixels within $1\sigma$ of the track and between $1\sigma$ and $2\sigma$,
at every threshold. It also places the same band shape at `n_null_bands` random
positions and orientations on the sky without the stream, to measure the
flagged densities the background alone produces in a stream-shaped region.

`stream_detection(results, THRESHOLD_GRID, at=(0.1, 0.5))` counts a stream as
detected when at least `min_pixels=20` pixels are flagged within $1\sigma$ and
`band_snr` $\ge$ `min_snr=2`. `band_snr` compares the band's flagged density
with the background bands' mean, in units of their scatter, never taken below
one pixel's Poisson noise. It reports the fraction of streams detected with a
Wilson interval, the median SNR, flagged counts and densities in the band and
side bands, and the background density. `plot_stream_detection` draws the
detected fraction against a parameter. The criterion and its choices are
explained in {doc}`../experiments/index`.

Results from `train_model.ipynb` §10-11 (batch Dice, batch 8; trained on
SB 31-34; 30 single-stream realizations per point; nside 512; threshold 0.5):

| SB | true pixels (mean) | $C$ | $F$ | $C/F$ | precision | background flagged, no stream |
|---|---|---|---|---|---|---|
| 30 | 569 | 0.92 | 7.7e-4 | 1200 | 0.97 | 0 |
| 31 | 525 | 0.90 | 1.0e-3 | 900 | 0.96 | 0 |
| 32 | 383 | 0.81 | 1.2e-3 | 655 | 0.91 | 0 |
| 33 | 206 | 0.13 | 4.3e-4 | 292 | 0.76 | 0 |
| 34 |  56 | 0 | -- | -- | -- | 0 |

Half of the true stream pixels are found down to **SB ≈ 32.4**. With the
stream removed, not one background pixel is flagged in any of the 210
realizations, so the background flagged on the stream skies is provoked by
the stream itself (pixels beside its track). At SB 34 the label still exists
(56 pixels on average) but the model does not respond.

An earlier version of this notebook trained with plain Dice at batch 2 found
more of the faint end at 0.5 (58% at SB 33) but flagged about 0.9% of the
background at every SB, even with no stream, on the same sky pixels from one
realization to the next. Per-window Dice gives no gradient on a window
without a stream, so nothing taught the model to predict low there. The
comparison that led to batch Dice is {doc}`../experiments/loss_selection`.

Current limits: one stream per sky. Multi-stream footprint injection is
the next step, and the functions above already take whatever the injector
returns. The scan is also one-dimensional (surface brightness only); a 2-D
version against distance modulus is planned.
