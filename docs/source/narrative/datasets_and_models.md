# Datasets and models

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
  `label_stack` stays in raw count units throughout, since the loss
  functions and evaluation metrics are defined against that literal scale.
- `StreamMapTransform` — composes optional normalization with optional
  augmentation (random 90° rotations, horizontal/vertical flips), applied
  identically to `map_stack`, `label_stack`, and `valid_mask` so all three
  stay in geometric lockstep.

No synthetic noise injection: this was in an earlier skeleton sketch, and
was deliberately removed. `map_stack` is count data with its own realistic
survey noise already baked in from injection, and the label is a literal
star count — adding an uncorrelated Gaussian noise model on top would teach
the network a noise model that doesn't match the real one.

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
classification), `"identity"` (unconstrained regression), `"softplus"`
(smooth non-negative regression — the natural fit for the current default
`label_policy="stream_count"`, a literal non-negative count).
`UNet.encoder()` exposes bottleneck features alone, for the Stage-2
embedding work described in {doc}`overview`.

## Losses ({py:mod}`streamgoggles.models.losses`)

All losses share one convention: `valid_mask` is a *true exclusion*, not a
downweight — invalid pixels contribute exactly zero to the loss, computed
via a masked sum divided by the valid pixel count (or, for
`WeightedMSELoss`, by the mask-restricted weight sum).

- `DiceLoss`, `FocalLoss`, `TverskyLoss`, `BCEWithLogitsLoss` — designed
  for (and still valid for) the binary/density label options.
- `MSELoss`, `WeightedMSELoss` — the primary pair for the current default
  `label_policy="stream_count"`. `WeightedMSELoss` weights each pixel's
  squared error by its own target count, so the (rare) high-count stream
  pixels aren't drowned out by the much more common near-zero background
  pixels.

`get_loss(name, **kwargs)` is a small factory over all six, by name.
