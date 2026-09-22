# Two streams in one window

**Question.** Every model is trained with exactly one stream per window. On
real data a window can hold two: neighbouring or crossing streams. Does a
model trained on single streams still find a stream when another one is
nearby, or does multi-stream training become necessary?

**Status.** Done, by evaluation alone (no retraining).

## Setup

Two of the selected models from {doc}`hyperparameters` (19200 training windows,
fixed decoy box, seeds 42 and 43), on the survey and region they were trained
on (LSST year 1, Dec -30), scored on a background they never saw.

**Paired design.** Stream A sits at the detection edge, SB 33, near the centre
of the study region, with a random orientation. In every condition A is the
same stream: the same realization, placement and survey noise. Only its
neighbour B changes:

- absent (A alone);
- parallel to A, offset across A's track by 1, 2 or 4 degrees;
- crossing A at its centre, at 60 degrees.

B is either brighter than A (SB 32) or as faint (SB 33). Both streams are 8
degrees long, 0.2 degrees wide, at distance modulus 16. That is 9 conditions,
each on the same 100 realizations of A, for each of the two models.

Because A is identical across conditions, the comparison is made realization
by realization: a stream A detected alone but missed with B is counted as
*lost*, the reverse as *gained*, and the significance comes from a McNemar
(sign) test on those counts. Detection uses the criterion of every experiment
(at least 20 flagged pixels within 1 sigma of the track and SNR at least 2),
with each model thresholded to flag 1e-3 of the stream-free sky.

## Result

**Statement.** A single-stream model is not confused by a neighbour 2 degrees
away or more, nor by a crossing stream. It **suppresses a faint stream running
parallel within about 1 degree of a brighter one**: stream A's detection falls
from 99.5% to 5.5%.

```{image} figures/two_streams/two_streams.png
:alt: Fraction of stream A detected, and its median flagged pixels, against the separation of a parallel neighbour B, for B brighter than A and as faint, with a crossing neighbour shown separately
:width: 100%
```

*Left: fraction of the 200 realizations of stream A detected (two models × 100
realizations), each model thresholded to flag 1e-3 of the stream-free sky; the
dotted line is A alone. Right: the median number of pixels flagged within
1 sigma of A's track at threshold 0.5 — the model's response to A itself,
before any detection test. Crosses: B crossing A at 60 degrees.*

| B | A detected | lost / gained vs alone | p | A's flagged pixels (median) |
|---|---|---|---|---|
| none (A alone) | 99.5% | — | — | 120 |
| parallel 1 deg, SB 32 | **5.5%** | 188 / 0 | < 1e-50 | **0** |
| parallel 2 deg, SB 32 | 98.5% | 2 / 0 | 0.5 | 98 |
| parallel 4 deg, SB 32 | 99.5% | 0 / 0 | 1 | 118 |
| crossing 60 deg, SB 32 | 100% | 0 / 1 | 1 | 91 |
| parallel 1 deg, SB 33 | **94.5%** | 10 / 0 | 0.002 | **55** |
| parallel 2 deg, SB 33 | 99.5% | 0 / 0 | 1 | 116 |
| parallel 4 deg, SB 33 | 99.5% | 0 / 0 | 1 | 117 |
| crossing 60 deg, SB 33 | 100% | 0 / 1 | 1 | 123 |

B itself is detected in every condition (91% or more), so the loss is always
the fainter, or equal, stream of the two.

**It is the model, not the detection test.** The density of flagged pixels in
stream-shaped bands on the stream-free sky is unchanged by B (about 2-3e-4 in
every condition), so the bar A has to clear is the same. What changes is the
model's response to A: with a brighter neighbour 1 degree away, A's flagged
pixels drop from a median of 120 to 0, and with an equally faint one to 55.
The network judges a pixel against its surroundings (its receptive field spans
about 3.5-4.5 degrees), and trained only on single streams it has learned that
a window holds one line: next to a brighter one, a faint parallel track is
treated as part of the surroundings. At 2 degrees a brighter neighbour still
costs A about a fifth of its flagged pixels, but not its detection. A crossing
stream, which only shares a small patch of sky with A, does not suppress it.

## Conclusion

- **Single-stream training is enough** for streams 2 degrees or more apart,
  and for crossing streams. The first application — recovering the known DES
  2018 streams one by one — can go ahead without multi-stream training.
- **The known limitation** is a faint stream running parallel within about
  1 degree of a brighter one. The same mechanism is likely to affect a faint
  stream next to any brighter structure (a brighter stream, a dwarf galaxy,
  the outskirts of the Magellanic Clouds), which is worth checking on real
  data.
- **If it matters**, the fix is cheap: inject a second stream in a fraction of
  the training windows, so the model sees neighbouring tracks. The evaluation
  code already builds multi-stream skies
  ({py:meth}`~streamgoggles.injector.StreamInjector.inject_streams_full_sky`),
  and this page's script can re-test a retrained model directly.

**Caveats.** Two models, one stream shape (0.2 degrees wide, 8 degrees long);
wider streams, which the next experiment trains on, will overlap sooner — two
1-degree-wide streams 1 degree apart touch. Separations between 1 and 2
degrees were not scanned, so the onset lies somewhere in that range.

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/experiments/two_streams/run.py      # scores 2 models x 9 conditions x 100 realizations (~30 min)
python scripts/experiments/two_streams/figures.py  # the figure and table above
```

`run.py` resumes where it stopped. Results are written to
`data/experiments/two_streams/`.
