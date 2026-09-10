# streamgoggles

`streamgoggles` is a machine-learning pipeline for detecting Milky Way
stellar streams in wide-field photometric survey data. It builds on
[`streamobs`](https://github.com) (synthetic survey observation and stream
injection) and [`ugali`](https://github.com/DarkEnergySurvey/ugali)
(isochrones and stellar population modeling) to generate realistic,
labeled training data, and on [`hyrax`](https://github.com) for eventual
production training orchestration.

## What the pipeline does

Given a footprint on the sky, `streamgoggles`:

1. Builds a synthetic (or real) **background** star catalog restricted to
   that footprint, applies photometric cuts, and caches per-filter,
   per-distance count maps.
2. **Injects** a synthetic stellar stream — realized from an isochrone
   population, placed along a random great circle, run through the survey's
   noise/detection model — and combines it with the cached background.
3. Runs one or more **matched filters** (a "good" isochrone-consistent
   filter, plus deliberately "bad" decoy filters) over the combined field at
   several trial distances, producing one input channel per
   `(filter, distance)` pair.
4. Trains a **U-Net** to recover the stream-only star count per channel —
   a background-cleaning (image regression) task — and evaluates it against
   a trivial threshold baseline.

See {doc}`narrative/overview` for the full picture, or jump straight to the
{doc}`api` reference.

```{toctree}
:maxdepth: 2
:caption: Guide

narrative/overview
narrative/data_generation
narrative/datasets_and_models
narrative/training_and_evaluation
narrative/notebooks
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
