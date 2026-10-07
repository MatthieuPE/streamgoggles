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
`des2018_training`); the trainings' configurations `hough/band2 residual des`
and `hough/band2s5 residual`.

**In short.** Trained only on simulations, the line network (four long
models per fold) finds **8 of the 14 DES 2018 streams along their known
tracks** — ATLAS, Chenab, Elqui, Jhelum, Phoenix, Tucana III, Turbio and
Turranburra — and 7 without knowing them, as in the sky search; the matched
filter's own line sums find 8 and 6, the two together 9 and 8. Of the
twelve streams in our data, only Aliqa Uma and Wambelong are found by
neither line search, by neither test: Aliqa Uma has 59% of its band under
the mask of Fornax's outskirts, Wambelong is at S/N 5.2. **The network finds
the real streams about as often as it finds their simulated copies at the
same strength** (6 found where its copies predict 4.1, of the nine streams
within the copies' range): the simulations predict how the model does on the
real sky. **The DES 2018 streams are 0.5 to 2 mag fainter in our data than
their copies at Table 1's surface brightness**, 33.0 to 35.5 on streamobs's
scale, where the training stopped at 34.5 — but training at their strength
makes the network less sensitive (8 streams along their tracks → 6). What
helps is the label: **a line taught only where the stream reaches S/N 5 in
the window, not 2, gives a better line network — 9 of the 14 along their
tracks and 7 without**, with two quick models per fold. With both
(the S/N-5 label at the streams' strength) the network gains near streams —
Wambelong, by a hair, on the sky — and loses far ones: no better overall.
Longer training (19,200 windows) buys nothing on the DES 2018 streams and
loses the short streams again, and more short streams in training make the
network worse at them. **Four quick S/N-5 models per fold are the best line
network so far: ten of the fourteen streams along their tracks, eight
without, nine in the sky search with the matched filter's line sums** — the
second pair of seeds confirms the first. **Segment lines** — the lines of
nine sub-windows beside the window's — help the matched filter's line search
(eight streams on the sky against six), not the network: its first pair's
nine on the sky was partly a good draw, and four segment models find what
four window-line models find. **Masking the bright dwarfs to 2° at most**,
where their stars end, instead of 12 half-light radii (Fornax's 4°) frees
Aliqa Uma at no cost: **eleven of the fourteen along their tracks**. A
shallower selection would trade streams, not add them: no single faint
limit serves every stream.

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

## Training for the DES 2018 streams

Two levers on the training, each against the same reference — the 2° line
model, two quick models per fold, its band label from S/N 2 in the window,
trained on 32 to 34.5 mag/arcsec² (`hough/band2 residual`) — on the copies
(fold-0 models on fold 1), the sky search and the real streams:

- **training at the streams' strength**: streams of 32.5 to 35.5
  mag/arcsec², where the real ones lie (`hough/band2 residual des`, the
  `population des2018` training set);
- **lines only where they show**: the band label from S/N 5 in the window
  instead of 2 (`hough/band2s5 residual`, the `band2s5` label); a stream
  that reaches S/N 5 in no placement is drawn again, so the network is never
  told "line" where the line cannot be seen. At 32-34.5, 38 of 40 drawn
  streams still find a placement where they do;
- **both**: the S/N-5 label at 32.5 to 35.5 (`hough/band2s5 residual des`),
  where 27 of 40 drawn streams find such a placement.

| two quick models per fold | lines from S/N 2, 32-34.5 | lines from S/N 2, 32.5-35.5 | **lines from S/N 5, 32-34.5** | lines from S/N 5, 32.5-35.5 |
|---|---|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.3 / 12.0 | 9.4 / 13.9 | 8.4 / **11.5** | **8.1** / 12.8 |
| copies: half found along the track, near / far | 7.1 / 8.6 | 8.0 / 12.3 | 6.6 / **8.3** | **6.1** / 8.9 |
| DES 2018 copies found without the track, near / far | **88%** / 89% | 85% / 74% | 85% / **94%** | **88%** / 71% |
| sky search: line network / both searches / either | 6 / 8 / 8 | 6 / 8 / 8 | **7** / 8 / 8 | **7** / 8 / **9** |
| **DES 2018 streams along their tracks / without** | 8 / 6 | 6 / 6 | **9 / 7** | 7 / **7** |

```{image} ../figures/line_model/des2018_training.png
:alt: Each DES 2018 stream found or not by the line network under the four trainings
:width: 100%
```

*Each DES 2018 stream, with its S/N in our data and its surface brightness
on streamobs's scale, found or not by the four trainings' line networks.
`run.py`, `des2018_training`.*

1. **Training at the streams' strength makes the network less sensitive,
   not more** — on the copies at every distance, and on the real streams
   along their tracks (Phoenix and Turbio lost). The per-pixel network's
   training down to 36 mag/arcsec² failed the same way
   ({doc}`../real_des/labels_normalization`).
2. **Teaching lines only where they show helps**: on the copies along the
   track (half found at S/N 6.6 and 8.3, near and far) and far without it
   (11.5), and on the real streams — **nine of the fourteen along their
   tracks** (Willka Yaku now found, *p* = 0.002) **and seven without them**
   (Phoenix back), the best of any line network so far, with two quick
   models per fold where the four long ones found eight and seven. The
   network alone now finds along the tracks what it found only together
   with the matched filter's line sums. The gains on the copies are of the
   order of the spread between seeds; on the real streams they are one
   stream each.
3. **The two levers read together**: a line the network cannot see is worse
   than no example. Training fainter adds such lines under the S/N-2 label
   — the level its stream-free windows reach in 1% of them fell from
   0.08-0.58 to 0.03-0.19 (fold 0), the network answering more softly
   everywhere; the S/N-5 label removes them. Training at the streams'
   strength with the S/N-5 label is the fourth column.
4. **Both together trade far streams for near ones**: the best on near
   copies (half found along the track at S/N 6.1, without it at 8.1), and
   on the sky the line network finds **Wambelong** — the first method to,
   though by a hair (its best line along the track clears the 1% level by
   0.001) — for nine with either search; but it loses the far copies (71%
   of the far DES 2018 copies at Table 1's surface brightness, 12.8 for
   half without the track) and, along their tracks, Turbio and Willka Yaku.
   No better overall than the S/N-5 label alone: the label is the lever, the
   training range is not.

## Longer training

The S/N-5 label (32-34.5) at the long tier: two models per fold trained on
19,200 windows instead of 4,800 (`hough/band2s5 residual long`), against its
two quick models per fold and the four long S/N-2 models per fold, the most
sensitive line model before. With the length scan's bright streams (32
mag/arcsec², m−M 17, 0.2° wide), found without their track, since one long
S/N-2 model per fold had lost the short ones ({doc}`line_followup`):

| | lines from S/N 5, 2 quick models | lines from S/N 5, 2 long models | lines from S/N 2, 4 long models |
|---|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.4 / **11.5** | **7.7** / 11.9 | 7.9 / 11.9 |
| copies: half found along the track, near / far | **6.6 / 8.3** | 7.7 / 8.9 | 7.1 / 8.9 |
| DES 2018 copies found without the track, near / far | 85% / **94%** | **90%** / 78% | 88% / 78% |
| bright streams found, 4° / 5° / 6° / 8° | **62% / 100% / 100% / 100%** | 0% / 12% / 38% / 100% | 0% / 12% / 38% / 100% |
| sky search: line network / both searches / either | 7 / 8 / 8 | 7 / 8 / 8 | 7 / 8 / 9 |
| **DES 2018 streams along their tracks / without** | **9** / 7 | 8 / 7 | 8 / 7 |

```{image} ../figures/line_model/des2018_long.png
:alt: Each DES 2018 stream found or not by the S/N-5 line network, quick and long, and by the four long S/N-2 models
:width: 100%
```

*Each DES 2018 stream found or not by the S/N-5 line network trained quick
and long, and by the four long S/N-2 models per fold. `run.py`,
`des2018_training(DES2018_LONG, "long")`.*

**Longer training buys nothing on the DES 2018 streams, and loses the short
streams again.** The long S/N-5 models find the same seven without their
tracks, one fewer along them (Willka Yaku), and of the bright streams 4-6°
long, at most 38% — where the quick models find 62-100%. Trained long, the
S/N-5 models answer as the S/N-2 ones do: different networks (their scores
differ by up to 0.35), the same answer on 99% of the copies. What the S/N-5
label gives, it gives at the quick tier: longer training undoes it,
whatever the label — so the short streams are lost by the training itself,
not by the label; likely because long streams dominate a training range of
4-30°, so that more training tunes the network to them.
Three more long models per fold are not worth training; the quick S/N-5
models are the line network to keep.

## More short streams in training

If the long training loses the short streams because a uniform 4-30° range
holds few of them (15% under 8°), training on more should keep them. The
S/N-5 model, two quick models per fold, with lengths log-uniform over the
same 4-30° (34% under 8°, median 11° against 17°; the `population short`
training set, `hough/band2s5 residual short`):

| two quick models per fold, lines from S/N 5 | lengths uniform | lengths log-uniform |
|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.4 / **11.5** | **8.3** / 11.9 |
| copies: half found along the track, near / far | **6.6** / 8.3 | 7.1 / **8.2** |
| DES 2018 copies found without the track, near / far | 85% / **94%** | **88%** / 76% |
| bright streams found without the track, 4° / 5° / 6° / 8° | **62% / 100% / 100% / 100%** | 0% / 25% / 50% / 100% |
| bright streams found along the track, 4° / 5° / 6° / 8° | **100% / 100% / 100% / 100%** | 25% / 88% / 100% / 100% |
| sky search: line network / both searches / either | **7** / 8 / 8 | 6 / 8 / 8 |
| **DES 2018 streams along their tracks / without** | **9 / 7** | 8 / 6 |

```{image} ../figures/line_model/des2018_short.png
:alt: Each DES 2018 stream found or not by the S/N-5 line network trained on uniform and on log-uniform lengths
:width: 85%
```

*Each DES 2018 stream found or not by the S/N-5 line network trained on
lengths uniform and log-uniform over 4-30°. `run.py`,
`des2018_training(DES2018_SHORT, "short")`.*

**More short streams in training make the network worse at short streams**,
not better: its median score on the bright 4-6° streams falls from 0.62 to
0.28 — the network's own answer, not its stream-free windows' level, which
hardly moves — and it loses Willka Yaku along its track and Phoenix without
it. So the short streams are not lost for want of examples. The line
network's answer to them swings widely from one training to the next — of
the bright 5° streams, found without their track 12% to 100% across the
trainings of this page, on two to four models per fold — so how much of the
quick S/N-5 models' lead is theirs, and how much a good draw of seeds, is
the open question.

## Four quick S/N-5 models per fold

Is the quick S/N-5 models' lead theirs, or a good draw of seeds? Two more
per fold (seeds 44 and 45), scored alone (`hough/band2s5 residual s44`) and
averaged with the first two (`hough/band2s5 residual x4`):

| lines from S/N 5, quick models | seeds 42-43 | seeds 44-45 | **all four** |
|---|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.4 / 11.5 | 7.9 / 12.0 | 8.0 / 11.5 |
| copies: half found along the track, near / far | 6.6 / 8.3 | 6.6 / **7.9** | **5.8** / 8.3 |
| DES 2018 copies found without the track, near / far | 85% / 94% | 88% / 93% | 88% / 94% |
| bright streams found without the track, 4° / 5° / 6° / 8° | 62% / 100% / 100% / 100% | 50% / 100% / 100% / 100% | 50% / 100% / 100% / 100% |
| sky search: line network / both searches / either | 7 / 8 / 8 | **8 / 9 / 9** | **8 / 9 / 9** |
| **DES 2018 streams along their tracks / without** | 9 / 7 | 9 / **8** | **10 / 8** |

```{image} ../figures/line_model/des2018_seeds.png
:alt: Each DES 2018 stream found or not by the quick S/N-5 line network, each pair of seeds and all four
:width: 100%
```

*Each DES 2018 stream found or not by the quick S/N-5 line network: the
first pair of models per fold, the second, and all four. `run.py`,
`des2018_training(DES2018_SEEDS, "seeds")`.*

**The lead holds**: the second pair finds the same nine streams along their
tracks and the short streams as well (bright 5-8° streams all found, 4° half
of them), so the S/N-5 label's gain is not a draw. **Four quick models per
fold are the best line network so far**: **ten of the fourteen DES 2018
streams along their tracks** — every stream in our data but Aliqa Uma, under
the Fornax mask, and Indus, which it finds without its track — and **eight
without**; in the sky search, eight for the network alone and **nine with
the matched filter's line sums**, both at half the rate. The tenth and the
eighth are **Wambelong** (S/N 5.2), found by a small margin — along its track
one stream-free window of 600 scores as high (*p* = 0.003), on the sky two
lines along it, 1.09 times the level, which chance gives in 1.5% of places —
but by the second pair and by its average with the first, and before by the
models trained at the streams' strength: a faint stream at the edge of what
the network sees, not a single model's accident. Along the track, half the
near copies are now found at S/N 5.8, near the matched filter's own band
test (5.2, {doc}`../real_des/labels_normalization`).

## Segment lines

A 4° stream fills a third of a window-long line, and the line's sum is
divided by the square root of all its pixels. The window's lines and those of
nine half-overlapping sub-windows of 48 pixels (5.5°, every 24 pixels) side
by side (`SegmentLines`, `streamgoggles.models.hough`): each sub-window's
lines are their own columns of the grid, with empty columns between parts so
that the line network's convolutions, the peak finding and the target never
mix two parts; a sub-window's lines hold a target only where the label runs
16 pixels along one. The matched filter's line search uses the same lines.
The S/N-5 model, two quick models per fold, same seeds
(`hough/band2s5 residual seg`):

| lines from S/N 5, two quick models per fold | window lines | window and sub-window lines |
|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.4 / 11.5 | **7.7 / 11.2** |
| copies: half found along the track, near / far | **6.6** / 8.3 | 7.3 / **8.2** |
| DES 2018 copies found without the track, near / far | 85% / **94%** | **90%** / 88% |
| bright streams found without the track, 4° / 5° / 6° / 8° | **62% / 100% / 100%** / 100% | 50% / 62% / 88% / 100% |
| sky search: line network / matched-filter lines / both / either | 7 / 6 / 8 / 8 | **9 / 8 / 9 / 11** |
| **DES 2018 streams along their tracks / without** | 9 / 7 | **10 / 8** |

```{image} ../figures/line_model/des2018_seg.png
:alt: Each DES 2018 stream found or not by the S/N-5 line network with window lines and with sub-window lines too
:width: 85%
```

*Each DES 2018 stream found or not by the S/N-5 line network with window
lines, and with the sub-windows' lines too. `run.py`,
`des2018_training(DES2018_SEG, "seg")`.*

1. **Segment lines help the search without the track** — with these two
   models; four, below, temper it: half the copies found at S/N 7.7 and
   11.2 (near and far) against 8.4 and 11.5, and on the sky **the line
   network alone finds nine** of the fourteen — Turbio
   (*p* = 0.04 against random tracks) and Turranburra (0.03) added to the
   seven — **the matched filter's segment lines eight** (Indus and Wambelong
   added), both at half the rate nine, **either at its own rate eleven**:
   all but Aliqa Uma, under the Fornax mask, and the two not in our data.
   Along their tracks, ten (Indus added).
2. **Not where expected**: the bright short streams are found no better
   without their track (5°: 62% against 100%), the near copies along the
   track a little worse. The gain is on the long, faint and wide streams —
   Turbio (15°, S/N 5.1), Turranburra (17°, 3.9), Indus (20°, 0.83° wide) —
   likely because their signal is patchy along the track: a sub-window line
   can pick the stretch that stands out, where a window line averages it
   with the stretch that does not.
3. **More lines on the sky**: 552 network lines and 1,620 matched-filter
   lines above their levels, against 223 and 430. The level is set so that 1%
   of stream-free windows hold a line, whatever their number of lines, so
   the extra lines come from windows holding structure — each structure now
   cut into several segments; the chance test against random tracks counts
   them. Turbio and Turranburra pass it narrowly.

### Four segment models per fold

Two more segment models per fold (seeds 44 and 45), alone
(`hough/band2s5 residual seg s44`) and with the first two
(`hough/band2s5 residual seg x4`), against the four window-line models per
fold:

| lines from S/N 5, quick models | window lines, all four | segment lines, seeds 42-43 | seeds 44-45 | **segment lines, all four** |
|---|---|---|---|---|
| copies: half found without the track, near / far (input S/N) | 8.0 / 11.5 | 7.7 / 11.2 | 8.1 / 11.2 | **7.7 / 11.1** |
| copies: half found along the track, near / far | **5.8 / 8.3** | 7.3 / 8.2 | 7.4 / 8.5 | 7.3 / 8.5 |
| bright streams found without the track, 4° / 5° / 6° / 8° | 50% / 100% / 100% / 100% | 50% / 62% / 88% / 100% | 62% / 88% / 100% / 100% | 50% / 62% / 100% / 100% |
| sky search: line network / both / either | 8 / 9 / 9 | 9 / 9 / 11 | 8 / 8 / 10 | 8 / 9 / **10** |
| **DES 2018 streams along their tracks / without** | **10 / 8** | 10 / 8 | 7 / 7 | 8 / 7 |

```{image} ../figures/line_model/des2018_segseeds.png
:alt: Each DES 2018 stream found or not by four window-line models per fold and by the segment-line models, each pair and all four
:width: 100%
```

*Each DES 2018 stream found or not by the four window-line models per fold
and by the segment-line models: each pair of seeds and all four. `run.py`,
`des2018_training(DES2018_SEG_SEEDS, "segseeds")`.*

**The first pair's nine on the sky was partly a good draw.** The second
pair finds eight on the sky and seven along the tracks; the four together
find eight on the sky and seven without the track — as the four window-line
models — and along the tracks eight against their ten. The test along the
track is harder for segment lines: a stream's own lines now include its
sub-windows' lines, so the stream-free windows have more chances to score as
high, and Turbio, Turranburra and Wambelong miss its level by a hair (*p* =
0.005-0.008 against 0.005). On the copies, the segment models find the far
ones a little more often without the track (half at 11.1 against 11.5), the
near ones a little less often along it. **For the line network, segment
lines are no better than window lines. For the matched filter's line search
they are**: in every segment run it finds eight streams on the sky against
six (Indus and Wambelong added; it does not depend on the network's seeds),
so that either search finds ten.

## The input: the dwarfs' mask

With the model at a plateau, the input. Aliqa Uma is in our data (S/N 7.9)
but 59% of its band lies under the search's mask of Fornax: 12 half-light
radii, 4.0°, where the band runs 2.4-7.8° from Fornax. Masking the bright
dwarfs keeps every line through their outskirts out of the search; how far
their stars reach in our matched filter decides how far that has to go:

```{image} ../figures/line_model/des2018_dwarf_profiles.png
:alt: Fornax's and Sculptor's matched-filter counts against the distance from their centres, with the two mask radii
:width: 100%
```

*Matched-filter counts in annuli around Fornax and Sculptor, over the 6-9°
ring, at three queried distances (Aliqa Uma's band and the other masked
objects left out). Grey, the object mask's radius; orange, the tight
mask's. `run.py`, `des2018_dwarf_profiles`.*

**Both dwarfs' stars stand out to 1.5°** — Fornax +5-33% at 1-1.5°,
Sculptor +12-18% there at m−M 18-19 — and nothing beyond but the
region's large-scale gradient of a few per cent. The **tight mask** masks
the bright dwarfs to 12 half-light radii but 2° at most
(`object_mask(max_radius_deg=...)`, `--mask tight`): Fornax from 4.0° to
2.0°, Sculptor from 2.2° to 2.0°, 40 deg² freed. The calibration mask, and
so the training and the stream-free windows, are unchanged. With the best
line network (four quick S/N-5 models per fold):

| search mask | object mask (12 half-light radii) | tight (at most 2°) |
|---|---|---|
| Aliqa Uma's band on searchable sky; its longest straight run in a window | 41%; 4.1° | **100%; 9.8°** |
| Aliqa Uma along its track: line network / matched-filter lines (*p*) | 0.20 / 0.042 | **0.003 / 0.002** |
| lines on the sky: network / matched filter; within 5° of Fornax | 210 / 430; 0 | 210 / 430; 0 |
| sky search: line network / both / either | 8 / 9 / 9 | 8 / 9 / 9 |
| **DES 2018 streams along their tracks: line network / both searches** | 10 / 9 | **11 / 10** |

**Freed, Aliqa Uma is found along its track by both searches, at no cost**:
the ring around Fornax adds no line to the sky search — its stars end at
1.5° — and the search finds the same streams without the tracks. Aliqa
Uma is not among them: at S/N 7.9 it is below what a search at 1% false
lines per window reaches. **Along their tracks the line network now finds
eleven of the fourteen DES 2018 streams — every stream in our data but
Indus**, which it finds without its track.

## The input: the selection's depth

Our selection reaches g and r = 24.5, a magnitude deeper than DES 2018's,
which takes Turranburra from S/N 8.6 to 3.9
({doc}`../real_des/des2018_reproduction`). Does a shallower limit help the
streams the search misses? Each stream's S/N in our matched filter at four
faint limits, from the inference catalogue's stars, without rebuilding any
map: selected by the matched filter at the queried distance nearest the
stream's own; their density across its DES 2018 track (on the search's
valid sky, away from the other DES 2018 streams), a quadratic fitted beyond
two widths, the stars within one width against it (`des2018_depth`):

| stream (its reference S/N) | g, r ≤ 23.0 | ≤ 23.5 | ≤ 24.0 | ≤ 24.5 (ours) | 23.5 against 24.5 |
|---|---|---|---|---|---|
| ATLAS (23.1) | 23.4 | 23.4 | **24.5** | 22.3 | +5% |
| Elqui (15.7) | 11.5 | 15.4 | **17.0** | 15.8 | −2% |
| Phoenix (12.9) | 16.2 | 17.1 | **17.3** | 15.1 | +14% |
| Jhelum (12.0) | 12.9 | 15.4 | 17.2 | **18.8** | −18% |
| Chenab (11.7) | 9.7 | 10.5 | **12.9** | 12.0 | −12% |
| Tucana III (11.1) | 15.3 | **18.1** | 16.2 | 16.8 | +8% |
| Indus (11.0) | 12.4 | 14.9 | 14.4 | **16.2** | −8% |
| Willka Yaku (8.7) | 1.6 | 3.6 | **5.7** | 5.5 | −35% |
| Aliqa Uma (7.9) | 5.6 | **6.7** | 6.3 | 4.9 | +36% |
| Wambelong (5.2) | 3.9 | 5.0 | 5.4 | **6.6** | −24% |
| Turbio (5.1) | 2.9 | 3.4 | 3.4 | **4.1** | −17% |
| Turranburra (3.9) | 5.5 | **6.0** | 3.1 | 3.2 | +86% |

*S/N of each stream's core in our matched filter, by the selection's faint
limit in g and r; the reference is `real_des.REAL_INPUT_SNR`'s, at the
stream's own distance and with its own masks and fit — this measurement
agrees with it for half the streams and not for Tucana III (its own dwarf
on the track), Indus, Jhelum, Willka Yaku or Aliqa Uma, so read the rows
across, limit against limit, not against the reference. `run.py`,
`des2018_depth`.*

**No single limit serves every stream.** A limit of 23.5 raises Turranburra
(+86%) and Aliqa Uma (+36%) and lowers Willka Yaku (−35%), Wambelong (−24%),
Jhelum (−18%) and Turbio (−17%); 24.0 is as mixed. The faint streams near
the search's limit split both ways, so a shallower selection would trade
streams, not add them, and rebuilding every sky, training and search on it is
not worth it. What could serve both is two selections side by side — a deep
and a shallow matched filter as separate input channels — a change of the
model's input rather than of its maps.

## What it means for the question

- **A network trained only on streamobs simulations finds, without being
  told where to look, eight of the fourteen DES 2018 streams** in our
  matched-filter maps, at one false line per hundred stream-free windows —
  four quick S/N-5 models per fold, with window or segment lines; with the
  matched filter's own line sums, nine, and ten with either search at its
  own rate (the matched filter's segment lines). **Along their known
  tracks, eleven** — the network alone, window lines, with the bright
  dwarfs masked to 2° at most: every stream in our data but Indus (found
  without its track).
- **What this input allows**: without the tracks, the streams missed are
  those our data hold at S/N 8 or less — Aliqa Uma (7.9), Wambelong (5.2)
  and Turbio (5.1) by one search or none, Turranburra (3.9) — and the two
  not in our data. Model changes have reached a plateau — one or two
  streams between pairs of seeds is the noise. What can still move the
  number is the input: the search's mask (done: Aliqa Uma, along its track);
  the selection's depth does not — a shallower limit raises Turranburra and
  Aliqa Uma but lowers Willka Yaku, Wambelong, Jhelum and Turbio — unless
  two selections, deep and shallow, are given to the model side by side.
- **The simulations predict the network's results on the real streams**, so
  the copies are a sound bench to improve it on — and what helped on the
  copies (the S/N-5 label) helped on the real streams.
- **What could move the number**, in order of expected gain: the input — our
  selection, a magnitude deeper than DES 2018's, takes Turranburra from S/N
  8.6 to 3.9 ({doc}`../real_des/des2018_reproduction`); the search's masks
  (Aliqa Uma under Fornax's 12 half-light radii); and the network's own
  sensitivity to short, narrow streams, which longer training loses
  whatever the label, and more short streams in training do not restore.

## Reproducing

```bash
python scripts/experiments/line_model/run.py des2018    # both configurations, ~25 min each, then the figures
python scripts/experiments/line_model/run.py des2018 --config "hough/band2 residual long x4"
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual" --seed 42   # 43; --train-sky fold1; ~37 min each, four at once; also "hough/band2 residual des", "hough/band2s5 residual des"
python scripts/experiments/line_model/run.py evaluate --config "hough/band2s5 residual"           # --train-sky fold1; --sets fainter
python scripts/experiments/line_model/run.py line-sky --config "hough/band2s5 residual"
python scripts/experiments/line_model/run.py des2018 --config "hough/band2s5 residual"            # and "hough/band2 residual"
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual seg" --seed 42   # 43; both folds; ~46 min each, four at once
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual" --seed 44   # 45; both folds; then evaluate / line-sky / des2018 the "hough/band2s5 residual x4" and "s44" ensembles
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual short" --seed 42   # 43; --train-sky fold1; ~33 min each, four at once
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual long" --seed 42   # 43; --train-sky fold1; ~2h15 each, four at once
python scripts/experiments/line_model/run.py evaluate --config "hough/band2s5 residual" --sets "length scan"   # and the long models'
```

The tight mask: `line-sky --config "hough/band2s5 residual x4" --mask tight` and
`des2018 --config "hough/band2s5 residual x4" --mask tight`; the profiles
`des2018_dwarf_profiles()`; the depths `des2018_depth()`. The comparisons are `des2018_training()`, `des2018_training(DES2018_LONG, "long")` and
`des2018_training(DES2018_SHORT, "short")`, `des2018_training(DES2018_SEEDS, "seeds")`,
`des2018_training(DES2018_SEG, "seg")`, `des2018_training(DES2018_SEG_SEEDS, "segseeds")` in `run.py`.
