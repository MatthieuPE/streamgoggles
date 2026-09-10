# Notebooks

Two executable notebooks in `notebooks/` walk through the pipeline
interactively. Both are built the same way: plain, editable config
dictionaries at the top, then the pipeline step by step, then the same
result built again with the minimal high-level call. Both were executed
for real (against the project's `streamml` conda environment) as part of
building them — not just written and assumed to work.

## `create_data.ipynb`

Builds **one** training sample two ways: step by step (study region →
cached background per named matched filter → realized stream → placed in
the footprint → injected through the survey model → windowed → per-channel
select/combine/finalize/crop, including the label) and then the same thing
via the single `StreamInjector.inject_single_stream` call. Also shows the
shape-compatible background-only case (`inject_background_only`). The best
starting point for understanding exactly what one `Sample` contains and
how it was built — see {doc}`data_generation` for the module-level version
of the same story.

## `train_model.ipynb`

Wires the rest of the pipeline together and runs one short, real training
loop end to end: `StreamMapDataset` (training + eval mode) →
`StreamMapTransform`/`RobustNormalizer` → `UNet` →
`models.losses.get_loss("weighted_mse")` → `PlainTrainer` → `evaluation/`
(`evaluate_on_grid`, `build_baseline`, `plot_recovery_vs_parameter`). It's
deliberately **smoke-scale** — small images, a small model, a handful of
epochs — to prove the whole pipeline trains and evaluates for real, not to
produce a scientifically tuned model. It plots a training/validation loss
curve, one prediction compared against its true label, and a completeness
(Dice) curve against the network's free richness parameter, with the k·σ
baseline overlaid for comparison. Scaling this up (bigger images, more
steps/epochs, a GPU device, and eventually the `training/hyrax_runner.py`
orchestration layer) is the natural next step once a scientifically
meaningful run is wanted.
