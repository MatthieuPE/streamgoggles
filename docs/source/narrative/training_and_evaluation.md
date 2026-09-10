# Training and evaluation

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

## Evaluation ({py:mod}`streamgoggles.evaluation`)

### Metrics ({py:mod}`streamgoggles.evaluation.metrics`)

`iou`, `dice`, `precision_recall`, `mse`, `correlation`, `weighted_recall`
— all pure-numpy, all `valid_mask`-aware (true exclusion, same convention
as the losses), all operating on whatever scale `pred`/`target` are
actually in (a `[0, 1]` probability for the binary/density label options,
or a literal star count for `"stream_count"`).

Every metric with an undefined ratio — `iou`/`dice` with nothing predicted
and nothing true, `precision` with no positive predictions, `correlation`
with fewer than two valid pixels or zero variance — returns `float("nan")`
rather than a guessed `0.0`/`1.0`, so it can never silently read as a real
score. Aggregating code is expected to `.dropna()` explicitly.

`weighted_recall` is worth calling out: it weights recall by the target's
own magnitude, so missing a pixel with many true stream stars costs more
than missing a nearly-empty one — a natural fit once the label is a literal
count rather than a binary flag.

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
