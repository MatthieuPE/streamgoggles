# Overview

## Two stages

The project is scoped into two stages, and only the first is implemented:

- **Stage 1 (implemented):** a background-cleaning network. Given a
  multi-channel image built from several matched filters at several trial
  distances, predict the true stream-only star count per channel — i.e.,
  learn to separate stream signal from background contamination.
- **Stage 2 (future work, not designed yet):** a detection network that
  takes Stage 1's cleaned output and predicts a per-pixel probability of
  stream presence, plus a `discovery/` module for embedding-based novel
  stream search. Both are explicitly out of scope for now — kept in mind,
  not built.

## Pipeline, end to end

```{mermaid}
flowchart LR
    A[StudyRegion] --> B[Background<br/>load_or_cache]
    B --> C[StreamInjector<br/>inject_single_stream]
    C --> D[Sample<br/>map_stack / label_stack / valid_mask]
    D --> E[StreamMapDataset]
    E --> F[StreamMapTransform<br/>normalize + augment]
    F --> G[UNet]
    G --> H[PlainTrainer]
    H --> I[evaluation/]
```

Each stage is a separate module, documented in its own guide page:

- {doc}`data_generation` — building one labeled training sample:
  background catalogs, matched filters, stream injection, windowing, and
  the resulting `Sample`.
- {doc}`datasets_and_models` — `StreamMapDataset` (on-the-fly training /
  fixed eval grid), preprocessing (`StreamMapTransform`), and the model
  (`UNet`, `models.losses`).
- {doc}`training_and_evaluation` — the training loop (`PlainTrainer`) and
  the evaluation toolkit (metrics, a k·σ baseline, per-parameter recovery
  curves).
- {doc}`notebooks` — two runnable notebooks that walk through data
  generation and a full (smoke-scale) training run.

## Design decisions worth knowing up front

A few choices run through the whole codebase and are easy to trip over if
you don't know them going in:

**Multi-channel input, one channel per `(filter, distance)` pair.**
Rather than a single matched filter at a single trial distance,
`streamgoggles` applies a *configurable set* of matched filters — including
at least one deliberately "bad" decoy filter (a plain color-magnitude box,
not isochrone-shaped) — at every trial distance. The network sees what
generic background contamination looks like *and* what a real
isochrone-consistent overdensity looks like, at the same distance, side by
side. Channels are ordered distance-major, filter-minor
(`channel_index = dist_idx * n_filters + filter_idx`), and the exact
mapping is always recorded explicitly in `Sample.metadata["channels"]` — no
code should ever hardcode or guess that ordering.

**The label is a literal star count, not a derived quantity.**
`label_stack[c]` is the true (noise-free) count of *stream-only* stars per
pixel that were selected by channel `c`'s specific matched filter at its
specific trial distance — background-excluded, and already restricted to
the sample's spatial window. Two alternatives were considered and rejected:
a binary/soft classification label (loses information the pipeline already
has for free) and a surface-brightness conversion (dominated by the
brightest few stars, and confounds the physical quantity with whatever
distance modulus the conversion assumes). A plain per-channel count is
directly comparable across filters and distances, and was already an
intermediate quantity the injection pipeline computed anyway. The default
`StreamInjector.label_policy` is `"stream_count"`; the earlier
binary/density/soft-distance labels (`rasterize.py`) remain implemented and
selectable, but are no longer the default path.

**Cuts and clipping are shared, not background-specific.**
Any photometric cut or magnitude clipping needs to apply identically to
background stars *and* injected stream stars — otherwise the network could
learn to distinguish them by a systematic cut artifact rather than real
astrophysics. `data_preparation.py` owns this logic precisely because it's
shared by both `background.py` and `injector.py`, not owned by either.

**Everything that touches real astrophysics or survey behavior is backed
by `streamobs`/`ugali`, wrapped, not reimplemented.** Isochrone population
synthesis, survey noise/detection modeling, matched-filter polygon
construction, and photometric error models all come from those packages;
`streamgoggles` orchestrates them and adds the training-data/ML-specific
layers (windowing, channel stacking, labeling, datasets, models).
