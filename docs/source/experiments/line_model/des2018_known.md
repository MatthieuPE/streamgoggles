# The DES 2018 streams where they are

The project asks whether streamobs simulations can train a model that finds
stellar streams in matched-filter maps in place of the by-eye inspection
that found the DES 2018 streams (Shipp et al. 2018). The benchmark is those
fourteen streams. The pages before searched the whole sky; this one asks,
stream by stream, what each method recovers **where DES 2018 found the
stream**, and for each miss, whether the data, the model or the simulation
is the cause.

Branch `des2018-recovery`; code `scripts/experiments/line_model/run.py`
(`des2018_known`, `des2018_strength`, `des2018_table`, `des2018_figures`,
`des2018_training`).

**In short.** Trained only on simulations, the line network (four long
models per fold) finds **8 of the 14 DES 2018 streams along their known
tracks** — ATLAS, Chenab, Elqui,
Jhelum, Phoenix, Tucana III, Turbio and Turranburra — and 7 without knowing
them, as in the sky search; the matched filter's own line sums find 8 and 6,
the two together 9 and 8. Of the twelve streams in our data, only Aliqa Uma
and Wambelong are found by neither line search, by neither test: Aliqa Uma
has 59% of its band under the mask of Fornax's outskirts, Wambelong is at
S/N 5.2. **The network finds the real streams about as often as it finds
their simulated copies at the same strength** (6 found where its copies
predict 4.1, of the nine streams within the copies' range): the simulations
predict how the model does on the real sky. What limits it is its
sensitivity from S/N about 12 down — where it finds half its copies or
fewer. **The DES 2018 streams are 0.5 to 2 mag fainter in our data than
their copies at Table 1's surface brightness**, 33.0 to 35.5 on streamobs's
scale, where the training stopped at 34.5; but **training at their strength
makes the network less sensitive, not more** (8 streams along their tracks
→ 6).

## The test

The copies' two tests ({doc}`window_level`), on the real streams in the real
sky, with the sky search's windows and models ({doc}`line_sky`):

- **each stream's band**: the sky within one width of its DES 2018 track
  (the paper's arc, as in {doc}`../real_des/des2018_reproduction`), at the
  queried distance nearest the stream's own;
- **its windows**: the sky search's windows (11°, half overlapping, the
  bright dwarfs and the globular clusters masked) that hold at least 2° of
  the band in a straight run, each scored by the models that never trained
  on its fold; the lines along the track in a window are the lines the band
  would be labelled with;
- **against stream-free sky**: the copies' 600 stream-free windows on the
  calibration sky of the window's fold, scored by the same models:
  - *along the known track*: in the window holding the longest straight run
    of the band, the best line along the track against the same lines in
    the stream-free windows; found when at most 1 in 201 score as high —
    the level of every known-track test so far;
  - *without the track*: a line along the track, in any of the stream's
    windows, above the level the best line of 1% of the stream-free windows
    reaches — the sky search's level;
  - *both line searches together*: each at half its rate, as in the sky
    search.

Two line networks are tested, both on the 2° label and S/N inputs: four
quick models per fold (`hough/band2 residual x4`) and four long ones
(`hough/band2 residual long x4`, the most sensitive on the copies,
{doc}`long_and_fits`). The per-pixel network is the first training's, with
its own known-track test ({doc}`../real_des/results`).

## Which method finds which stream

```{image} ../figures/line_model/des2018_methods.png
:alt: The fourteen DES 2018 streams, strongest first, and which method finds each along its known track and without it
:width: 100%
```

*Each DES 2018 stream, by its S/N in our data (the stars within one width of
its track against the density profile around it,
{doc}`../real_des/des2018_reproduction`): filled, found; open, not. Left,
along the known track; right, without it. `run.py`, `des2018_figures`.*

| stream | S/N in our data | band on searchable sky | longest straight run in a window | line network (4 long), *p* along the track | matched-filter lines, *p* | found without the track |
|---|---|---|---|---|---|---|
| ATLAS | 23.1 | 100% | 12.3° | **0.002** | **0.002** | both |
| Elqui | 15.7 | 91% | 9.9° | **0.002** | **0.002** | both |
| Phoenix | 12.9 | 98% | 8.2° | **0.002** | **0.002** | both |
| Jhelum | 12.0 | 99% | 12.5° | **0.002** | **0.002** | network |
| Chenab | 11.7 | 100% | 12.3° | **0.002** | **0.002** | both |
| Tucana III | 11.1 | 100% | 3.7° | **0.002** | **0.002** | matched filter (and the 4 quick models) |
| Indus | 11.0 | 100% | 14.2° | 0.022 | 0.30 | network |
| Willka Yaku | 8.7 | 100% | 6.6° | 0.008 | **0.002** | matched filter |
| Aliqa Uma | 7.9 | **41%** | 4.1° | 0.30 | 0.042 | — |
| Wambelong | 5.2 | 100% | 13.7° | 0.010 | 0.060 | — |
| Turbio | 5.1 | 100% | 10.2° | **0.002** | **0.002** | — |
| Turranburra | 3.9 | 79% | 8.9° | **0.002** | 0.075 | network |
| Molonglo | 1.1 | 63% | 4.8° | 0.88 | 0.39 | — |
| Ravi | −0.2 | 100% | 11.9° | 0.053 | 0.17 | — |

*p: the share of the 600 stream-free windows whose same lines score as high
(plus one, over 601); 0.002 is none of them, bold is found (at most 1 in
201).*

1. **Along their known tracks, the line network finds eight streams and the
   matched filter's line sums eight**, not the same eight: the network
   finds Turranburra, which the matched filter does not; the matched filter
   Willka Yaku, which the network misses narrowly (4 stream-free windows
   score as high). Together, each at half its rate, they find nine. The
   per-pixel network found five ({doc}`../real_des/results`).
2. **Without the track, the network finds seven, the matched filter six,
   the two together eight** — the sky search's numbers ({doc}`line_sky`,
   {doc}`long_and_fits`): a line along a stream counts the same whether the
   search matches it to the track afterwards or the test looks along it.
3. **Turbio is found along its track by both, at S/N 5.1, but not without
   it**: what the data hold is there, but below what a search can claim at
   1% false lines per window.
4. **Indus is found without its track but not along it**: the test along
   the track uses the window holding its longest run (14°), where its best
   line stands out less than in another of its nine windows. Wide (0.83°)
   and long, it is the one stream the choice of window decides.
5. **Ravi and Molonglo, not in our data, are found by nothing**: the tests
   do not find streams that are not there.

## Why the misses

### The streams are fainter in our data than their copies

The copies of {doc}`window_level` are each DES 2018 stream simulated with its
Table 1 parameters, its surface brightness included, at random places of the
calibration sky. At that surface brightness **most copies are much stronger
in our data than the streams themselves** — Chenab's copies reach S/N 26
where Chenab has 12, Tucana III's 29 where it has 11, Indus's 78 where it
has 11. The copies' S/N falls as their flux, 0.40 dex a magnitude (the
copies 1 and 1.5 mag fainter, `fainter` set), so each stream's S/N in our
data reads off as a surface brightness on the simulation's scale:

| stream | Table 1 | copies' S/N at Table 1 | S/N in our data | fainter by | on streamobs's scale |
|---|---|---|---|---|---|
| ATLAS | 33.0 | 22.0 | 23.1 | −0.05 | 32.95 |
| Tucana III | 32.0 | 29.3 | 11.1 | 1.04 | 33.04 |
| Phoenix | 32.6 | 19.7 | 12.9 | 0.46 | 33.06 |
| Jhelum | 33.3 | 18.3 | 12.0 | 0.46 | 33.76 |
| Willka Yaku | 32.9 | 23.9 | 8.7 | 1.07 | 33.97 |
| Indus | 31.9 | 77.7 | 11.0 | 2.14 | 34.04 |
| Wambelong | 33.7 | 8.6 | 5.2 | 0.53 | 34.23 |
| Turbio | 32.6 | 21.8 | 5.1 | 1.63 | 34.23 |
| Aliqa Uma | 33.8 | 11.8 | 7.9 | 0.47 | 34.27 |
| Elqui | 34.3 | 18.3 | 15.7 | 0.16 | 34.46 |
| Chenab | 34.1 | 25.7 | 11.7 | 0.87 | 34.97 |
| Turranburra | 34.0 | 16.2 | 3.9 | 1.52 | 35.52 |

*Surface brightness in mag/arcsec²; `run.py`, `des2018_strength`. Molonglo
and Ravi are not in our data.*

The twelve streams lie at **33.0 to 35.5 on the simulation's scale, median
34.1**. The line models were trained on 32 to 34.5 (`population`): nearly
two fifths of that range brighter than ATLAS, the strongest DES 2018 stream
in our data, and Chenab and Turranburra beyond its faint end. Table 1's surface brightness is the
paper's own measurement, with its selection, on DES Y3; why the same number
gives streamobs streams 0.5-2 mag stronger in our selection is not
settled here, but the training range does not have to wait for it.

### Each stream against its copies at its strength

```{image} ../figures/line_model/des2018_copies.png
:alt: Each DES 2018 stream in our data, found or not, against the share of its copies found at its strength
:width: 100%
```

*Each stream in our data at its S/N, against the share of its own copies
found at that S/N — read between its copies at Table 1's surface
brightness and 1 and 1.5 mag fainter, in log S/N; filled, the real stream
found. A point high and open would be a stream the simulation says the
method should find, missed.*

Summed over the nine streams within their copies' range (all but ATLAS,
brighter than its copies, and Indus and Turbio, fainter), the copies predict
how many the real sky gives:

| nine streams within the copies' range | copies predict | found |
|---|---|---|
| line network, along the track | 4.1 | 6 |
| matched-filter lines, along the track | 7.1 | 6 |
| line network, without it | 3.5 | 5 |
| matched-filter lines, without it | 4.7 | 5 |

**The real streams are found as often as their copies at the same strength
predict** — the network even slightly more often. The one stream that
three quarters or more of its copies say should be found, and is not, is
**Aliqa Uma**, by the matched filter along its track (76% of its copies):
59% of its band lies under the
mask of Fornax's outskirts (12 half-light radii, `object_mask`), and its
longest straight run in a window is 4° of its 10°. The copies, placed on
fully valid sky, do not see that mask.

So the misses are of three kinds:

- **not in the data**: Ravi and Molonglo;
- **masked**: Aliqa Uma, by the search's mask of Fornax;
- **too faint for the method**: Willka Yaku (8.7) for the network, whose
  copies at that strength it finds 11% of the time; Wambelong (5.2) for
  both; Turbio (5.1) and Turranburra (3.9) without the track. From S/N
  about 12 down, the network finds half of a stream's copies or fewer — the
  sensitivity limit of {doc}`window_level` — and the training gave it few
  streams there.

## Training at the streams' strength

The direct lever: train where the real streams are. The same 2° line model,
two quick models per fold, trained on streams of 32.5 to 35.5 mag/arcsec²
(`hough/band2 residual des`, the `population des2018` training set) against
the same models trained on 32 to 34.5 (`hough/band2 residual`), on the
copies (fold-0 models on fold 1), the sky search and the real streams:

| two quick models per fold | trained on 32-34.5 | trained on 32.5-35.5 |
|---|---|---|
| copies: half found without the track, near / far (input S/N) | **8.3 / 12.0** | 9.4 / 13.9 |
| copies: half found along the track, near / far | **7.1 / 8.6** | 8.0 / 12.3 |
| DES 2018 copies found without the track, near / far | **88% / 89%** | 85% / 74% |
| sky search: line network / both searches / either | 6 / 8 / 8 | 6 / 8 / 8 |
| DES 2018 streams along their tracks / without | **8** / 6 | 6 / 6 |

```{image} ../figures/line_model/des2018_training.png
:alt: Each DES 2018 stream found or not by the line network trained on 32-34.5 and on 32.5-35.5 mag/arcsec2
:width: 70%
```

*Each DES 2018 stream, with its S/N in our data and its surface brightness
on streamobs's scale, found or not by the two trainings' line networks.
`run.py`, `des2018_training`.*

**Training at the streams' strength does not help: it makes the network
less sensitive** — on the copies at every distance, and on the real streams
along their tracks (Phoenix and Turbio lost), for the same six without them
and the same sky. The per-pixel network's training down to 36 mag/arcsec²
failed the same way ({doc}`../real_des/labels_normalization`). Two readings,
not yet told apart:

- **the label**: the band label calls a stream a line wherever its band
  reaches S/N 2 in the window, so a fainter training set shows the network
  more lines it cannot see; taught to answer "line" where nothing shows, it
  answers more softly everywhere — the level its stream-free windows reach
  in 1% of them fell from 0.08-0.58 to 0.03-0.19 (fold 0);
- **the budget**: harder examples may need more than the quick tier's
  4,800 windows.

## What it means for the question

- **A network trained only on streamobs simulations finds, without being
  told where to look, six to seven of the fourteen DES 2018 streams** in our
  matched-filter maps, at one false line per hundred stream-free windows;
  with the matched filter's own line sums, eight. Along their known tracks,
  eight; nine with the matched filter's line sums.
- **Eight is about what this input allows at that false-alarm rate**: the
  streams it misses without their track are those our data hold at S/N
  5 or less (Wambelong, Turbio, Turranburra), Aliqa Uma under the Fornax
  mask, and the two that are not in our data.
- **The simulations predict the network's results on the real streams**, so
  the copies are a sound bench to improve it on.
- **What could move the number**, in order of expected gain: the input — our
  selection, a magnitude deeper than DES 2018's, takes Turranburra from S/N
  8.6 to 3.9 ({doc}`../real_des/des2018_reproduction`); the search's masks
  (Aliqa Uma under Fornax's 12 half-light radii); and the network's own
  sensitivity to short, narrow streams (Willka Yaku, Tucana III), where the
  matched filter's line sums still do better — the label's S/N threshold is
  the next lever there.

## Reproducing

```bash
python scripts/experiments/line_model/run.py des2018    # both configurations, ~25 min each, then the figures
python scripts/experiments/line_model/run.py des2018 --config "hough/band2 residual long x4"
python scripts/experiments/line_model/run.py train --config "hough/band2 residual des" --seed 42   # 43; --train-sky fold1; ~37 min each, four at once
python scripts/experiments/line_model/run.py evaluate --config "hough/band2 residual des"           # --train-sky fold1; --sets fainter
python scripts/experiments/line_model/run.py line-sky --config "hough/band2 residual des"
python scripts/experiments/line_model/run.py des2018 --config "hough/band2 residual des"            # and "hough/band2 residual"
```

The comparison of the two trainings is `des2018_training()` in `run.py`.
