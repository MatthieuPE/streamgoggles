# Datasets and models

```{note}
This page assumes familiarity with U-Nets, loss functions, and the
segmentation metrics it names (Dice, MSE, and so on) — see
{doc}`ml_concepts` first if any of those need a primer.
```

## `StreamMapDataset` ({py:mod}`streamgoggles.datasets.stream_map_dataset`)

A duck-typed torch `Dataset` (only `__len__`/`__getitem__`, no
`torch.utils.data.Dataset` subclassing, no top-level `torch` import) with
two modes:

- **Training** (`eval_mode=False`): every `__getitem__` call generates a
  fresh sample — a real injected stream, or (with probability
  `StreamConfig.background_fraction`) a pure-background window. `idx` has
  no stable identity here; `__len__` (`steps_per_epoch`) is a nominal
  per-epoch count for `DataLoader`, not a bound on how many distinct
  samples exist.
- **Eval** (`eval_mode=True`): walks a fixed `EvalGrid` (built from
  `StreamConfig` if not given explicitly). Each grid point has its own
  fixed seed, so `__getitem__(idx)` always returns the same sample —
  always persisted through the `SimulationStore`, regardless of
  `StreamConfig.persist`, since an evaluation set has to be reproducible
  across runs.

### Exactly how many samples, and how each one is built

Both modes ultimately call `StreamInjector.inject_single_stream(params,
rng)` exactly **once** per sample — one call realizes the stream
population, places it, samples **one** window (`windows.
sample_stream_window`), injects survey noise, then loops over every
`(distance, filter)` channel *reusing that same window and realization* to
build each channel's crop (see {doc}`data_generation`). So for a single
sample, window selection happens once, not once per channel — every
channel is a different color-magnitude/distance cut of the same underlying
window and stars, not an independently re-windowed draw. "A training
sample" always means: one stream population, one placement (random sky
position + orientation), one window, one set of survey noise draws —
projected into as many `(filter, distance)` channels as the config asks
for, all channels sharing that same underlying realization.

**One training step, concretely.** Say `richness` is `DISCRETE`
`[31, 32, 33, 34]` and `DataLoader` is about to build the next batch.
For each item in that batch, `StreamMapDataset.__getitem__` does, in
order: (1) spawn a fresh child RNG, never used before and never reused
again; (2) with probability `background_fraction`, generate a pure-
background window instead and stop here; otherwise (3) draw *every* free
parameter independently from that RNG — for `richness`, one of the 4
values, each equally likely, independent of what any other batch or epoch
drew; (4) call `injector.inject_single_stream(params, rng)`, which
realizes a brand-new stream population at that richness, places it at a
random sky position and orientation inside the footprint, rejection-samples
one window containing enough of it, injects survey noise, and crops every
channel from that one window/realization. The resulting `map_stack`/
`label_stack` pair becomes one row of one batch, contributes to exactly
one gradient update, and is then discarded — nothing about it is stored or
referenced again.

**Training: is a given input reused, across epochs or otherwise? No.**
`steps_per_epoch` is a nominal per-`DataLoader`-epoch count, not the
number of distinct samples that exist — `StreamMapDataset.rng` (an
`np.random.Generator` created once, when the dataset object is
constructed) is never reset between epochs, and every `__getitem__` call
spawns a fresh, never-repeated child generator from it
(`Generator.spawn(1)`) before sampling that call's own random `params` and
handing them to `inject_single_stream`. So the true number of distinct
training samples generated across a full `trainer.train(...)` run is
`epochs * steps_per_epoch` (times `batch_size`, since a "step" here means
one `__getitem__` call, and `DataLoader` calls `__getitem__` once per item
in a batch) — e.g. `train_model.ipynb`'s `epochs=40`,
`steps_per_epoch=30`, `batch_size=2` means **1200 independent,
never-repeated draws** over the course of training, not 30 samples
replayed 40 times, and not one fixed "training set" the way a typical
image dataset (loaded once, iterated over repeatedly) would be — this
dataset behaves like an infinite simulator you draw fresh from every step,
closer in spirit to online/streaming training than to epoch-over-a-fixed-
corpus training. Nothing is persisted to disk by default in training mode
(`StreamConfig.persist=False` unless set) — every draw really is generated
from scratch, window included. With multiple `DISCRETE` values, a fixed
total sample budget splits across them *on average* (each draw picks one
value independently, uniformly), not per-value — directly relevant to why
a training budget that used to fully serve one fixed richness now only
serves each of 4 values a quarter as often on average, see
{doc}`training_and_evaluation` (PLAN.md §6.13).

### Eval grid: one fixed realization per point — a real limitation, not just a design note

`config.build_eval_grid` enumerates the **Cartesian product** of every
non-`FIXED` parameter's candidate values (`UNIFORM`/`LOG_UNIFORM`:
`n_points_per_range` equally-spaced points, default 5; `DISCRETE`: every
value, or up to `n_points_discrete` if capped) — e.g. a single free
`DISCRETE` `richness` spec with 4 values produces exactly 4 grid points,
one per value (`train_model.ipynb`'s `[31, 32, 33, 34]` scan). Each point
is assigned its own fixed integer seed, drawn deterministically from a
single master `seed` (default 42) in enumeration order — so re-building
the same `StreamConfig` always produces the same points *and* the same
seeds. `StreamMapDataset._get_eval_sample` then calls
`inject_single_stream(params, np.random.default_rng(seed))` — **exactly
one, fully deterministic realization per grid point**: one specific
placement, one specific window (position *and* orientation), one specific
draw of survey noise, every time. This is generated once and then cached
(`SimulationStore.get_or_generate` loads the saved sample on every later
call instead of regenerating), so a given grid point's realization is
fixed not just within one run but across every epoch of it.

**This means a per-richness metric (e.g. "Dice at surface_brightness=34")
is a measurement on a sample size of exactly one placement/window/
orientation, not an average over the many realizations a stream with those
physical parameters could actually produce.** It does *not* marginalize
over window position, orientation, or noise the way the metric's name
("Dice at SB 34") suggests it might — a real gap between what the number
sounds like it means and what it actually measures. This was a deliberate
trade-off, not an oversight: fixing the realization per grid point makes
validation-loss/metric curves *within* one training run directly
comparable epoch to epoch (a Dice value going up or down between epoch 10
and epoch 20 reflects the model improving or not, not a new random window
also changing underneath it) — resampling a new window every epoch would
make that curve noisy for a different reason, confounding "is the model
learning" with "did this epoch happen to get an easier window."

**The real cost: a result like PLAN.md §6.13's "SB 34 recovers 0.0 Dice"
cannot be read as "the network genuinely cannot detect streams this
faint" — it could equally be "this one particular window/orientation
happened to be an unusually hard instance of SB 34."** Both are consistent
with the same observed number, and in that specific case the second turned
out to be closer to the truth (see `train_model.ipynb` §8).

### Replicates: several realizations per grid point

`build_eval_grid(config, n_replicates=k)` is the fix, and it is
implemented: each parameter combination is emitted **k times with k
different seeds**, so one "grid point" becomes k independent realizations
— different placement, window, orientation and survey noise — of the same
physical parameters. A per-parameter metric then becomes a mean and
spread instead of one arbitrary draw of it, which is what a number like
"Dice at SB 34" should have meant all along.

Replicates are expanded *inline* — the same point dict repeated, each
entry still carrying exactly one seed — rather than nesting a seed list
per point. That deliberately keeps `EvalGrid`'s "one index → one sample"
contract intact, so `StreamMapDataset.__getitem__`, `evaluate_on_grid`'s
DataFrame shape, and everything downstream needed no changes at all:
replicate entries are simply more rows, identified by
`EvalGrid.replicates` (and by a `replicate` column in `evaluate_on_grid`'s
output, read from `Sample.metadata["replicate"]`). The default is
`n_replicates=1`, i.e. exactly the historical behavior.

{py:func}`~streamgoggles.evaluation.completeness_purity.aggregate_over_replicates`
reduces those rows to `<metric>_mean`/`<metric>_std`/`<metric>_n` per
parameter combination. The `_n` column is not bookkeeping: NaN metrics
(the undefined-ratio convention above) drop out per metric, so a mean over
2 of 5 replicates is a materially different claim from a mean over 5 of 5,
and the column is what tells them apart.

```{warning}
A real trap this had to handle explicitly: `SimulationStore` addresses
samples by their parameter dict, so replicates of one combination —
identical params, different seeds — would all collide on a *single* cache
entry and silently collapse k realizations back into one. The replicate
index is therefore part of the store's addressing key while deliberately
staying out of the params handed to `inject_single_stream` (which only
ever takes real stream parameters). The same hazard applies to *two
different grids sharing one store*: a 1-replicate grid and a k-replicate
grid both have a "replicate 0" whose addressing key is the bare params,
but they generally want different seeds there — give them separate stores
(as `train_model.ipynb` does) rather than letting whichever generates
first silently win.
```

### Parallel sample generation: what each worker actually does

Sample generation is CPU-bound and single-threaded, so
`DataLoader(num_workers=N)` is the cheapest real speedup available
(0.119 → 0.049 s/sample measured with 4 workers). What follows is exactly
what is shared between workers and what each one produces on its own,
measured by instrumenting real workers rather than inferred:

**Shared — identical in every worker:**

- **The background.** The catalog and its cached per-`(filter, distance)`
  HEALPix maps are built **once, in the parent process, before any worker
  exists**, and every worker then holds the same content (copy-on-write
  under Linux `fork`, an unpickled copy under macOS/Windows `spawn`).
  Verified: all 4 workers reported a byte-identical background map
  (checksum `7889d88afb`, 375,677 stars). Workers do **not** regenerate a
  background, and the "each matched filter touches the background exactly
  once" guarantee is unaffected — filters were already applied upstream.
- **The matched filters, injector configuration and normalizer**, for the
  same reason.

**Independent — drawn fresh by each worker, for every sample:**

- **The RNG stream** (each worker takes its own deterministic child — see
  the warning below),
- **the free parameters** (e.g. which `richness` this sample gets),
- **the stream realization** (a new population drawn from the isochrone),
- **the placement** — sky position *and* orientation,
- **the window**, and
- **the survey noise / detection draw**.

So a worker is not producing a variation of a shared stream: it runs the
whole of `inject_single_stream` itself and emits a complete, independent
sample. In the same measurement, 8 samples across 4 workers gave **8
distinct windows and 8 distinct maps**, with richness varying freely
between them, against that one shared background.

```{warning}
This independence only holds because it was made to. Workers receive a
copy of the dataset carrying `rng` in the parent's exact state, so by
default every worker spawns the same children and generates **identical
samples** — measured at **2 unique out of 8** with `num_workers=4`, with
no error and no slowdown to reveal it. `StreamMapDataset` gives each
worker its own deterministic child stream to prevent this; outside a
worker the RNG is left untouched, so single-process runs stay
reproducible.
```

**Eval mode is unaffected by worker count.** Eval samples take their seed
from the grid point (`EvalGrid.seeds[idx]`), never from the dataset RNG,
so the same index yields the same sample at any `num_workers` — metrics
stay comparable across runs that parallelize differently.

**Per-worker caches.** The isochrone caches
({doc}`data_generation`) are per-process, so each worker builds the
isochrone once: 4 constructions instead of 1, then hits thereafter
(measured: `misses=1` in every worker). Negligible against what they save.

**Choosing the count:** `default_num_workers()` deliberately does not take
the whole machine, and respects a cluster allocation rather than the
physical node — `SLURM_CPUS_PER_TASK`, then `os.sched_getaffinity` (the
cpuset the process is actually confined to), with `os.cpu_count()` only as
a last resort, so a 4-CPU job on a 128-core node gets a couple of workers
rather than 126. `STREAMGOGGLES_NUM_WORKERS` overrides it, including `0`.

```{important}
Pass `persistent_workers=True` whenever `num_workers > 0`. Worker startup
costs ~12s under `spawn` (macOS/Windows, where the dataset is pickled and
streamobs re-imported per worker) and ~0 under `fork` (Linux); without
persistence that is paid **every epoch**, which made a first attempt
**21x slower** than using no workers at all.

Also note that anything reachable from the dataset must be **importable**,
not just picklable, under `spawn`: a class defined in a notebook cell
fails with `AttributeError: Can't get attribute ... on <module
'__main__'>`. That is why `TransformedDataset` lives in this package
rather than in the notebook that uses it.
```

### `DataLoader` and `stream_map_collate_fn`

Wrapping this dataset in a real `torch.utils.data.DataLoader` needs a
custom `collate_fn` whenever `background_fraction > 0`: torch's default
collate function requires every sample's `"params"` dict to share the same
keys across a batch, but a background-only sample's `params` is `{}` while
a stream sample's carries `richness`/`morphology`/etc. — a batch mixing the
two crashes with a bare `KeyError`. `stream_map_collate_fn` fixes this: it
batches `map_stack`/`label_stack`/`valid_mask` via torch's own
`default_collate`, and leaves `params`/`metadata` as plain per-sample lists
instead of merging them (which is all `PlainTrainer` ever needs from a
batch anyway).

## Preprocessing ({py:mod}`streamgoggles.datasets.transforms`)

- `RobustNormalizer` — per-channel `(x - mean) / std`, fit once on pooled
  training data, computed over *valid* pixels only so the fixed invalid-fill
  value never skews the statistics. Applied to `map_stack` only —
  `label_stack` is never normalized, whatever scale it's actually in (a
  literal count under `label_policy="stream_count"`, already bounded `{0,
  1}` under `label_policy="stream_detection"`), since the loss functions and
  evaluation metrics are defined against that literal label scale.
- `StreamMapTransform` — composes optional normalization with optional
  augmentation (random 90° rotations, horizontal/vertical flips), applied
  identically to `map_stack`, `label_stack`, and `valid_mask` so all three
  stay in geometric lockstep.

No synthetic noise injection: this was in an earlier skeleton sketch, and
was deliberately removed. `map_stack` is count data with its own realistic
survey noise already baked in from injection, and the label is derived
directly from the same underlying stream-only count (literal under
`label_policy="stream_count"`, thresholded under `label_policy=
"stream_detection"`) — adding an uncorrelated Gaussian noise model on top
would teach the network a noise model that doesn't match the real one.

## The model ({py:mod}`streamgoggles.models.unet`)

A standard encoder–decoder U-Net with skip connections
({py:class}`~streamgoggles.models.unet.UNet`). Two details specific to this
project:

- **Input channels** are `n_distances × n_filters` (see
  {doc}`data_generation`'s channel-ordering convention), not just
  `n_distances` — one channel per `(filter, distance)` pair.
- **Small, often non-power-of-2 image sizes.** `PixelizationSpec.
  image_size_pix` is commonly 15×15–60×60, not the large power-of-2 inputs
  U-Nets are usually sized for. Downsampling uses max-pooling with
  `ceil_mode=True` so odd sizes round up instead of vanishing; upsampling
  resizes to the *exact* spatial size of the matching skip connection
  (`F.interpolate(..., size=skip.shape[-2:])`) rather than assuming a fixed
  2× scale factor, so skip concatenation never needs cropping or padding at
  any depth.

Three output heads, tied to the label/loss in use: `"sigmoid"` (binary
classification — paired with `label_policy="stream_detection"`, this
project's current default, see {doc}`ml_concepts`), `"identity"`
(unconstrained regression), `"softplus"` (smooth non-negative regression —
the natural fit for `label_policy="stream_count"`'s literal, non-negative
count, still available but no longer the default). `UNet.encoder()` exposes
bottleneck features alone, for the Stage-2 embedding work described in
{doc}`overview`.

**The output is always a continuous value, whichever head is used —
never literally boolean, even for `head="sigmoid"`.** A sigmoid-activated
output is a real number in `(0, 1)` at every pixel; nothing about the head,
or about training against a binary target, collapses it to exactly `0` or
`1`. This matters because `label_policy="stream_detection"`'s *label* is a
hard 0/1 (see below) — it would be easy to assume that makes the *model's
prediction* boolean too, but it doesn't: the label only shapes what the
continuous output comes to mean (roughly, the estimated probability that
the true detection condition holds, given the noisy input). Thresholding
into a hard decision is something a caller does afterward (exactly what
`evaluation.metrics`'s `threshold` parameter is for), not something baked
into the model.

### Predictions are never automatically masked — you always must mask them

`map_stack`/`label_stack` are exactly `0.0` outside `valid_mask` at every
stage of data preparation ({doc}`data_generation`'s "decision 22" fill
convention) — this is a real, verified invariant, not just documented
intent (see {py:mod}`streamgoggles.datasets.stream_map_dataset`'s tests).
`RobustNormalizer` preserves it too: it only ever writes to
`out[c][valid_mask]`, so invalid pixels pass through **completely
untouched by normalization** — they stay at their raw `0.0` fill value even
inside an otherwise mean-0/std-1 normalized channel. That's a subtle trap:
a raw `0.0` sitting among z-scored data (mean 0, std 1) can visually read
as just another unremarkable value near the channel's mean, rather than as
the obviously-invalid sentinel it actually is — it's easy to eyeball a plot
and not notice anything is wrong.

The bigger version of the same trap is the model's **output**. `UNet.
forward()` is a plain per-pixel convolutional pass with no awareness of
`valid_mask` at all — nothing about the architecture or the masked losses
in `models.losses` constrains what it predicts at invalid spatial
positions. Masked losses only exclude invalid pixels from the *training*
gradient; they do nothing to the model's behavior at *inference* time, so
`model(x)` routinely produces nonzero, physically meaningless values
outside the footprint. This is expected, not a bug — but it means **any**
code that plots, thresholds, or otherwise interprets a prediction must
intersect it with `valid_mask` first, the same way
`evaluation.metrics`/`models.losses` already do internally.
`notebooks/train_model.ipynb` §6 masks every panel, including its residual
plot (`np.where(valid_mask, image, np.nan)` + `cmap.set_bad`), for exactly
this reason — an earlier unmasked version of that plot was genuinely
confusing for this exact reason.

## Losses ({py:mod}`streamgoggles.models.losses`)

All losses share one convention: `valid_mask` is a *true exclusion*, not a
downweight — invalid pixels contribute exactly zero to the loss, computed
via a masked sum divided by the valid pixel count (or, for
`WeightedMSELoss`, by the mask-restricted weight sum).

- `DiceLoss`, `FocalLoss`, `TverskyLoss`, `BCEWithLogitsLoss` — the primary
  choices for the current default `label_policy="stream_detection"` (a
  bounded `{0, 1}` per-pixel target; `DiceLoss` + `head="sigmoid"` is what
  `train_model.ipynb` actually uses), and also still valid for the earlier
  binary/density label options.
- `MSELoss`, `WeightedMSELoss` — the primary pair for `label_policy=
  "stream_count"` (a literal, unbounded count), still available but no
  longer the default — see {doc}`ml_concepts` for why: MSE on that
  heavy-tailed target reliably localized streams but badly under-recovered
  their peak amplitude, which is what motivated the `stream_detection`
  pivot in the first place. `WeightedMSELoss` weights each pixel's squared
  error by its own target count, so the (rare) high-count stream pixels
  aren't drowned out by the much more common near-zero background pixels —
  see the real limitation below if reaching for it.

**A real limitation, found while tuning `train_model.ipynb`, not a
hypothetical:** `WeightedMSELoss`'s weight normalization
(`normalize_weight=True`) is computed over the **entire batch tensor**, not
per sample. For a batch that mixes very different richnesses (e.g. one
bright stream alongside two fainter ones), the bright sample's pixels can
end up owning nearly the whole batch's weight sum, leaving the other
samples in that batch with almost no gradient signal at all. This wasn't
severe enough to matter at this project's earlier, more modest richness
range, but became the dominant effect once streams got bright enough for
per-pixel counts to span orders of magnitude within one batch — at that
point, `models.losses.MSELoss` on a `log1p`-transformed target (see
{doc}`notebooks`'s description of `train_model.ipynb`) turned out to behave
far better than reweighting the raw counts. Not changed in `models/
losses.py` itself (out of scope for the notebook tuning that surfaced it) —
flagged here for whoever reaches for `WeightedMSELoss` next on a
similarly-skewed target.

`get_loss(name, **kwargs)` is a small factory over all six, by name.
