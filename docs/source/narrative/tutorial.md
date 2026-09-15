# Tutorial: from config to detection

A task-oriented walkthrough, complementing {doc}`notebooks`'s
cell-by-cell tour: how do I fix the parameters I want, control how much
data gets generated, train a model, and then actually use it to look for
streams across a real HEALPix footprint? Every snippet here is the same
code the two executable notebooks run for real — this page explains the
*choices*, the notebooks are where to run it.

## 1. Fixing vs. scanning parameters

Every stream parameter (`richness`, `width`, `length`, `morphology`,
`distance_modulus`, `age`, `z`, ...) is a
{py:class}`~streamgoggles.config.ParameterSpec`, one of four kinds
({py:class}`~streamgoggles.config.DistributionType`):

```python
from streamgoggles.config import DistributionType, ParameterSpec

# Fixed: always this exact value.
ParameterSpec(name="width", dist_type=DistributionType.FIXED, value=0.2)

# Discrete: one of a fixed, named set of values.
ParameterSpec(
    name="richness", dist_type=DistributionType.DISCRETE, values=[30.0, 31.5, 33.0]
)

# Uniform / log-uniform: any value in a continuous range.
ParameterSpec(
    name="distance_modulus", dist_type=DistributionType.UNIFORM, min_val=15.0, max_val=18.0
)
```

A {py:class}`~streamgoggles.config.StreamConfig` bundles a `{name:
ParameterSpec}` dict plus `background_fraction` (see below),
`richness_kind` (which unit `"richness"` is expressed in — `"nstars"`,
`"mass"`, or `"surface_brightness"`), and `persist`:

```python
from streamgoggles.config import StreamConfig

stream_config = StreamConfig(
    params={...},  # one ParameterSpec per name, as above
    background_fraction=0.2,
    richness_kind="surface_brightness",
    persist=False,
)
```

**"Fixed" and "scanned" both feed the exact same `StreamInjector.
inject_single_stream` call** — a `FIXED` spec always resolves to its one
value; a `DISCRETE`/`UNIFORM`/`LOG_UNIFORM` spec resolves differently
depending on mode: sampled randomly during training
({py:class}`~streamgoggles.datasets.stream_map_dataset.StreamMapDataset`,
`eval_mode=False`), or enumerated exhaustively (every combination, a
Cartesian product across every non-fixed parameter) for evaluation
({py:func}`~streamgoggles.config.build_eval_grid`, used automatically in
`eval_mode=True`). A config with **every** parameter fixed has nothing to
build an eval grid from — `build_eval_grid` raises `ValueError` in that
case; give at least one parameter a `DISCRETE`/`UNIFORM`/`LOG_UNIFORM`
spec if you want an evaluation sweep over it.

Grid size grows as the **product** of each free parameter's own point
count, so it's easy to make accidentally huge: two `UNIFORM` parameters at
the default `n_points_per_range=5` is already 25 points; three is 125.
`n_points_discrete` caps how many values a `DISCRETE` spec contributes
(subsampled evenly via `np.linspace` over its index, not randomly) if you
want to scan a rich named set without enumerating all of it.

## 2. Sizing how much data gets generated

Two different things control "how much data," for two different modes:

- **Training** (`eval_mode=False`): every `__getitem__` call generates a
  *fresh* sample on the fly — there's no fixed dataset size. `steps_per_epoch`
  is purely a nominal per-epoch count for `DataLoader`/`PlainTrainer`'s
  epoch loop, not a cap on distinct samples; `epochs × steps_per_epoch` is
  the real total number of generated training samples across a run.
  `background_fraction` (on `StreamConfig`) is the fraction of those draws
  that are pure background (`inject_background_only` — no stream, no
  label computation, cheap) rather than a real injected stream.
- **Evaluation** (`eval_mode=True`): a *fixed*, reproducible grid — see
  §1's eval-grid sizing — walked once, each point persisted via
  `SimulationStore` (always, regardless of `StreamConfig.persist`) so
  repeated access is deterministic and, after the first pass, cheap.

**Generation cost scales directly with stream richness**, not with
`steps_per_epoch`/`epochs` themselves — a real, measured effect, not a
guess: injecting a stream with ~11K stars takes a small fraction of a
second; ~1.1M stars (surface brightness fainter than ~28 mag/arcsec² at a
typical geometry) takes several seconds. Before committing to a training
budget, it's worth timing `injector.inject_single_stream(params, rng)`
once at the *richest* point your config can draw — that number, times
`epochs × steps_per_epoch`, is roughly your wall-clock training time (see
{doc}`notebooks`'s description of `train_model.ipynb`'s own tuning for a
concrete example: ~450 optimizer steps at a deliberately-bounded richness
range took on the order of ten minutes).

## 3. Training

```python
from streamgoggles.models.unet import UNet
from streamgoggles.models.losses import get_loss
from streamgoggles.training.plain_runner import PlainTrainer
from streamgoggles.storage import ModelStore

model = UNet(in_channels=n_channels, out_channels=n_channels, base_width=12, depth=2, head="sigmoid")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
trainer = PlainTrainer(model=model, optimizer=optimizer, loss_fn=get_loss("dice"))

model_store = ModelStore("data/models")  # a real, persistent directory -- not a tempdir, if you want the checkpoint to outlive this process
result = trainer.train(
    train_dl, val_dl=val_dl, epochs=30,
    model_store=model_store,
    config={"model_config": dict(in_channels=n_channels, out_channels=n_channels, base_width=12, depth=2, head="sigmoid")},
)
```

`in_channels`/`out_channels` must equal `n_distances × n_filters` (see
{doc}`data_generation`'s channel-ordering convention) — this has to match
whatever `matched_filter_cfg`/`filters_cfg` built the training data with.
`config["model_config"]` is not optional bookkeeping: it's the exact kwargs
`ModelStore.load_model(UNet, params)` will later call `UNet(**config)`
with, so a real (non-tempdir) `model_store` path is what makes a trained
model reusable in a later process — see {doc}`training_and_evaluation`
for `PlainTrainer`'s full contract (checkpointing, `save_every`, mixed
precision).

`head="sigmoid"` + `loss_name="dice"` is this project's current default
pairing, for the current default `label_policy="stream_detection"` (a
bounded `{0, 1}` per-pixel target — see {doc}`data_generation`). If you
specifically want `label_policy="stream_count"`'s literal count back
instead, use `head="softplus"` + `get_loss("mse")`/`get_loss("weighted_mse")`
— but be aware its dynamic range can span orders of magnitude for bright,
rich streams, in which case plain raw-count regression trains poorly; the
`log1p(count)` training-target technique described in {doc}`notebooks` (an
earlier version of `train_model.ipynb`) is the fix for that specific case,
not a `models.losses`/`models.unet` change. `stream_detection`'s bounded
target doesn't need it.

## 4. Detecting streams across a HEALPix map

Once a model is trained, running it over an entire survey footprint
(rather than one hand-placed window) uses
{py:func}`~streamgoggles.windows.tile_footprint` — a regular grid of
non-overlapping `Window`s covering `background.footprint`, built
specifically for this "full-survey inference" case (as opposed to the
random/stream-anchored windows `sample_random_window`/
`sample_stream_window` use for generating *labeled training* samples).
`tile_size_deg` should match `pix.image_size_pix[0] * pix.pixel_scale_deg`
so tiles actually abut the way `pix` will later crop them; a
`StudyRegion` only a little larger than one tile (a good choice for
*training* data, since one window is enough there) yields only one or two
tiles here — a real scan wants a `StudyRegion` sized for the area you
actually want covered.

```python
from streamgoggles.windows import tile_footprint
from streamgoggles.injector import inject_background_only

tiles = tile_footprint(
    background.footprint, pix.nside, tile_size_deg=pix.image_size_pix[0] * pix.pixel_scale_deg
)

model_for_eval.eval()
detections = []
with torch.no_grad():
    for window in tiles:
        sample = inject_background_only(background, window, pix)  # no stream, no label -- real/unlabeled sky
        if not sample.valid_mask.any():
            continue
        map_stack = normalizer(sample.map_stack, sample.valid_mask)  # the SAME fitted normalizer training used
        pred = model_for_eval(torch.from_numpy(map_stack).unsqueeze(0))[0].numpy()
        score = pred[:, sample.valid_mask].sum()  # or .max(), or a proper metric -- see below
        detections.append({"ra": window.center_ra, "dec": window.center_dec, "score": score})
```

`inject_background_only` — the same function {doc}`data_generation`
describes as the shape-compatible *negative* case for training
(`background_fraction > 0`) — turns out to be exactly the right tool for
an unlabeled sky tile too: it builds `map_stack` per `(filter, distance)`
channel the same way a training sample's does, with no injected stream and
no label, from whatever real sky sits inside `window`. **Don't rebuild
that channel loop by hand from `crop_window` directly** — a real mistake
made writing this tutorial: `Background.finalized_map_full_dict[filter][dm]`
is `None` whenever `finalize_cfg` is disabled (the common case; see
{doc}`data_generation`'s `finalize_full` note) rather than falling back to
the raw map automatically, and `inject_background_only` already contains
the correct raw/finalized fallback so callers don't have to duplicate it.
Reuse the same fitted `RobustNormalizer` training used — never re-fit one
per tile, or each tile's normalization becomes inconsistent with what the
model actually learned.

A few things worth being deliberate about:

- **`pred` still needs masking.** Exactly the trap described in
  {doc}`datasets_and_models`: `model_for_eval`'s output has no awareness
  of `valid_mask` on its own. `score = pred[:, valid_mask].sum()` above
  already accounts for this; don't drop the mask when trying other
  aggregations.
- **Turning a per-tile score into a detection decision** is exactly what
  {py:func}`~streamgoggles.evaluation.baseline_threshold.build_baseline`
  and {py:func}`~streamgoggles.evaluation.completeness_purity.compute_completeness_purity`
  are for on labeled data (a real stream at a known location); on real
  survey tiles with no label, the natural analogue is a plain threshold on
  `score` (or a per-pixel threshold on `pred` itself, then inspecting
  connected regions) — genuinely open-ended, and exactly the boundary
  where this pipeline's Stage 1 (this page) hands off to the Stage 2
  detection network described in {doc}`overview` (not built yet).
- **This loop is intentionally plain Python**, not vectorized across
  tiles — `tile_footprint` can return anywhere from a handful of tiles
  (a small `StudyRegion`) to many thousands (a full survey), and a real
  full-survey run is exactly the kind of job
  `training/hyrax_runner.py`/a GPU device is for, not a single-process
  CPU loop like the rest of this tutorial.
