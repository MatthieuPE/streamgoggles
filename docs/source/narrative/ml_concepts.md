# Machine learning concepts, explained

The other guide pages assume familiarity with the ML vocabulary they use —
U-Net, Dice, MSE, and so on. This page fills that in: what each concept
*is*, in general, before the other pages get into why this project uses it
a particular way. Read this first if any of those terms are unfamiliar;
skip it (or use it as a glossary) otherwise.

## Neural networks, briefly

A neural network is a function built out of layers, each one a simple,
differentiable operation (usually a linear transformation followed by a
nonlinearity) applied to the previous layer's output. Stacked deep enough,
these simple operations can approximate very complex functions.
**Training** means adjusting the numbers inside those layers (the
*weights*) so the network's output gets closer to some target, using
**gradient descent**: compute how wrong the current output is (the
**loss**), compute how a tiny nudge to each weight would change that loss
(via **backpropagation**, the calculus chain rule applied layer by layer),
and nudge every weight a small step in the direction that reduces the
loss. Do this repeatedly, on many examples, and the network gradually gets
better at the task.

A few terms that recur throughout this project's code and docs:

- **Epoch**: one full pass through however many training examples a run is
  configured to use per pass.
- **Batch**: a small group of examples processed together in one gradient
  step (rather than one at a time — faster, and averages out noisy
  individual-example gradients).
- **Optimizer step**: one weight update, computed from one batch. This is
  the real unit of "how much has the network actually learned" — an epoch
  with a huge batch and a tiny dataset might only be a handful of
  optimizer steps, which matters a lot for how much training budget is
  actually needed (see {doc}`tutorial`).
- **Learning rate**: how big each weight-update step is. Too large and
  training can diverge or oscillate; too small and it takes forever to
  get anywhere.
- **Overfitting** and the **train/validation split**: a network can, given
  enough capacity and training, memorize its exact training examples
  rather than learning a generalizable pattern — it'll then look great on
  the data it trained on and fail on anything new. The standard defense is
  to hold out a separate **validation set** the network never trains on,
  and watch its loss on that set too: if training loss keeps improving
  while validation loss stalls or gets worse, that's overfitting.

## Convolutional neural networks

A **convolutional** layer slides a small learned filter (a few pixels
across) over an image, producing a new image (a **feature map**) where
each output pixel summarizes a small neighborhood of the input. Early
layers tend to learn simple filters (edges, gradients); stacking many
convolutional layers lets later ones respond to increasingly complex,
larger-scale patterns built from simpler ones underneath — this is what
lets a CNN recognize shapes and structures, not just individual pixel
values. A **channel** is one such feature map; a convolutional layer
typically takes several input channels and produces several output
channels, each channel a different learned "detector."

**Pooling** (or a strided convolution) reduces an image's spatial size
(e.g. halves both dimensions), trading spatial resolution for a larger
effective receptive field per output pixel — deeper layers "see" more of
the original image per pixel, at coarser resolution. This is how a
network builds up broad, image-level context, not just local detail.

## U-Net

A **U-Net** ([Ronneberger, Fischer & Brox
2015](https://arxiv.org/abs/1505.04597), originally for biomedical image
segmentation) is a convolutional architecture for **dense prediction**
tasks — where the output is itself an image, the same spatial size as the
input, not a single label or a handful of numbers. This project's task
(predict something *per pixel* — a detection probability, or a star count)
is exactly that shape, which is why a U-Net rather than a
classification-style CNN (which collapses the whole image down to one
label or a handful of numbers).

It has three parts, and the "U" shape in the usual diagram names the
first two:

- **Encoder** (the left side of the "U"): a sequence of convolution +
  pooling stages, each one halving the spatial resolution while
  increasing the channel count. This builds up increasingly abstract,
  large-context features at increasingly coarse resolution — by the
  bottleneck (the bottom of the "U"), the network has a compressed,
  high-level summary of the whole image, but has lost fine spatial detail
  along the way.
- **Decoder** (the right side): a mirrored sequence of upsampling +
  convolution stages, each one doubling resolution back up, eventually
  returning to the original input size.
- **Skip connections** (the horizontal lines across the "U"): this is
  U-Net's key idea. At each decoder stage, the corresponding
  same-resolution encoder feature map gets concatenated in directly,
  alongside the upsampled decoder features. Without this, the decoder
  would have to reconstruct fine spatial detail purely from the
  bottleneck's compressed, low-resolution summary — which loses exactly
  the pixel-precise boundaries a segmentation task needs. Skip connections
  let the network combine broad context (from the deep path) with
  precise, local detail (carried across directly from the matching
  encoder stage) at every resolution.

The very end of the network (the **head**) maps the decoder's final
feature maps down to the actual number of output channels wanted, then
optionally applies an **activation function** to constrain the output's
range — see "Activation functions" below.

Here's the shape of this project's `UNet` (`models/unet.py`) specifically,
at its default `depth=4` (the number of encoder/decoder stages — this
project's actual runs often use a smaller `depth=2`, e.g. `train_model.ipynb`,
for a faster smoke-scale training loop; the diagram below just picks 4 to
show the pattern clearly):

```{mermaid}
flowchart TB
    IN["Input\n(B, in_channels, H, W)"] --> E1
    subgraph ENC["Encoder"]
        direction TB
        E1["ConvBlock 1\nwidth = base_width"] --> P1["MaxPool /2"]
        P1 --> E2["ConvBlock 2\nwidth = base_width x 2"] --> P2["MaxPool /2"]
        P2 --> E3["ConvBlock 3\nwidth = base_width x 4"] --> P3["MaxPool /2"]
        P3 --> E4["ConvBlock 4\nwidth = base_width x 8"] --> P4["MaxPool /2"]
    end
    P4 --> BOT["Bottleneck ConvBlock\nwidth = base_width x 16"]
    subgraph DEC["Decoder"]
        direction BT
        BOT --> U4["Upsample x2"] --> D4["ConvBlock\n(+ skip from E4)"]
        D4 --> U3["Upsample x2"] --> D3["ConvBlock\n(+ skip from E3)"]
        D3 --> U2["Upsample x2"] --> D2["ConvBlock\n(+ skip from E2)"]
        D2 --> U1["Upsample x2"] --> D1["ConvBlock\n(+ skip from E1)"]
    end
    D1 --> HEAD["1x1 conv -> out_channels\n+ head activation\n(sigmoid / identity / softplus)"]
    HEAD --> OUT["Output\n(B, out_channels, H, W)"]

    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
    E4 -. skip .-> D4
```

**Input and output are exactly the same spatial size (`H, W`).** This is
guaranteed by construction, for *any* `H, W` (not just powers of 2): pooling
uses `ceil_mode=True` (so an odd spatial size rounds up instead of
vanishing), and each decoder stage upsamples back to its matching skip
connection's *exact* recorded size via `F.interpolate(..., size=skip.shape[-2:])`
rather than a fixed scale factor — so there's never a size mismatch to crop
or pad around, at any `depth`. What changes between input and output is only
the **channel count**: `in_channels` is `n_distances * n_filters` (one
channel per `(matched filter, trial distance)` pair — see
{doc}`data_generation`), while `out_channels` is typically the same number
(one prediction per input channel, matching `label_stack`'s own shape) but
doesn't have to be — see `datasets_and_models` for how this project sets it.
The **output is a plain per-pixel convolutional result, with no built-in
awareness of `valid_mask`**: pixels outside the survey footprint still get
*some* (physically meaningless) predicted value, and must be masked out by
the caller before plotting or thresholding — exactly like the loss and
evaluation metrics already do internally (see the masking note in
{doc}`datasets_and_models`).

See {doc}`datasets_and_models` for exactly how this project's `UNet`
implementation is configured in practice (`base_width`, `depth`, `head`)
and used.

## Activation functions (the model's "head")

An activation function is a fixed, non-learned nonlinearity applied to a
layer's output — inside the network, one is needed between every
convolution or the whole stack would collapse into one big linear
function; at the *output*, one is chosen to constrain the prediction to
whatever range makes sense for the task:

- **Sigmoid**: squashes any real number into `(0, 1)` — a probability.
  Used when predicting "is a stream present here," a yes/no question per
  pixel.
- **Identity** (no activation): the raw, unconstrained output — used for
  plain regression, where any real number (including negative) is a
  valid prediction.
- **Softplus**: `log(1 + exp(x))`, a smooth version of "clip to
  non-negative" (unlike a hard clip, it's differentiable everywhere,
  which matters for gradient descent). Used when predicting a quantity
  that's physically never negative — a star count, here — without
  forcing it into `(0, 1)` the way sigmoid would.

## Normalization

Neural networks generally train better when their inputs are on a
consistent, well-behaved numeric scale (roughly centered on 0, unit
variance) rather than arbitrary raw units — very large or very
differently-scaled inputs across channels can make gradient descent slow
or unstable. **Standardization** (or "z-scoring") is the usual fix:
`(x - mean) / std`, computed from representative data, applied to every
input before it reaches the network. See {doc}`datasets_and_models` for
this project's specific approach (`RobustNormalizer`) and a real trap it
has: the normalization is only ever applied to *valid* pixels, so an
invalid pixel's raw fill value sits, untouched, inside an otherwise
normalized channel.

## Loss functions and metrics for image prediction

A **loss function** is what gradient descent actually minimizes during
training — it must be differentiable. A **metric** is anything used to
*evaluate* a trained model's quality — it doesn't need to be
differentiable, since no gradient is taken through it, so metrics can be
more directly interpretable than the loss that produced the model. The
same underlying idea (e.g. Dice) sometimes shows up as both a
loss (`models.losses.DiceLoss`) and a metric
(`evaluation.metrics.dice`) in this project, computed slightly
differently for that reason — the loss stays differentiable (soft,
continuous values), the metric is free to threshold/binarize.

### Regression: is the predicted *value* correct?

- **MSE (mean squared error)**: the average of
  $(\text{prediction} - \text{true})^2$ over all pixels. The standard loss for regression — predicting a
  continuous quantity (like a star count) rather than a class. Squaring
  the error means large mistakes are penalized much more than small ones.
- **Weighted MSE**: MSE where each pixel's squared error is scaled by a
  weight (often the target value itself) before averaging — used to make
  the loss care more about pixels with a large true value (e.g. a bright
  stream core) than the much more numerous near-zero background pixels,
  which would otherwise dominate a plain average just by sheer numbers.
  See {doc}`datasets_and_models` for a real limitation of this technique
  found in this project.
- **Correlation (Pearson correlation coefficient)**: measures whether
  predicted and true values move together (regardless of their absolute
  scale) — $+1$ means perfectly correlated, $0$ means unrelated, $-1$
  means perfectly inversely correlated. Useful for asking "does the model
  get the *pattern* right," separately from "does it get the exact
  magnitude right" (which MSE already answers).

### Segmentation: is the predicted *region* correct?

These all compare two regions — typically "predicted positive pixels" vs.
"true positive pixels," found by thresholding a probability or count at
some cutoff — and ask how well they overlap. They matter most when
*most* pixels are the uninteresting "negative" class (background) and
only a few are the interesting "positive" one (stream) — a plain
per-pixel accuracy would look great on a model that just predicts
"background everywhere," which these are specifically designed not to
reward:

- **IoU (Intersection over Union)**, also called the **Jaccard index**:
  $\lvert P \cap T\rvert \,/\, \lvert P \cup T\rvert$, with $P$ the predicted
  and $T$ the true region — the overlap between the two
  regions, divided by their combined extent. `1` means a perfect match,
  `0` means no overlap at all.
- **Dice coefficient** (also called the F1 score in this binary-overlap
  context): $2\lvert P \cap T\rvert \,/\, \big(\lvert P\rvert + \lvert T\rvert\big)$ — closely
  related to IoU (always `>=` it), and the more common choice in medical/
  scientific image segmentation specifically. As a *loss* (`DiceLoss`),
  it can be computed directly on continuous, un-thresholded probabilities
  (a "soft" version), which is what makes it usable in gradient descent
  at all — the *metric* version in this project's `evaluation.metrics`
  thresholds first, for an interpretable, literal overlap fraction.
- **Precision and recall**: split "how good is the overlap" into two
  separate questions with a real tradeoff between them.
  **Precision** $= \mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ — of everything the model flagged as
  positive, what fraction was actually right? Low precision means lots
  of false alarms. **Recall** $= \mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ — of everything that was
  actually positive, what fraction did the model catch? Low recall means
  lots of missed detections. (`TP`/`FP`/`FN` = true/false positive/
  negative pixel counts.) A model can trivially get perfect recall by
  flagging everything positive (at the cost of terrible precision), or
  perfect precision by flagging almost nothing (at the cost of terrible
  recall) — neither alone is a meaningful measure of quality, which is
  why they're normally reported together, and why Dice/IoU (which
  combine both) are often preferred as a single summary number.
- **Focal loss**: a modification of cross-entropy (see below) that
  down-weights pixels the model is already confidently getting right,
  so training focuses its gradient signal on the pixels it's still
  getting wrong — designed for the same "overwhelmed by easy negatives"
  problem Dice/weighted losses address, from a different angle (reshaping
  the loss curve near confident-correct predictions, rather than
  reweighting by region size).
- **Tversky loss**: a generalization of Dice that lets false positives
  and false negatives be weighted independently (Dice implicitly weighs
  them equally) — turn one knob up to make the model more conservative
  (fewer false alarms, at the cost of missing more true positives), or
  the other to make it more aggressive.
- **BCE (binary cross-entropy)**: the standard loss for a per-pixel
  yes/no probability prediction, derived from the negative log-likelihood
  of the true label under the predicted probability — mathematically, the
  "correct" loss for a probabilistic binary classifier in the same sense
  MSE is the "correct" loss for Gaussian-noise regression.
  `BCEWithLogitsLoss` combines the sigmoid activation and the BCE formula
  into one numerically stable computation, rather than applying sigmoid
  first and risking `log(0)`.

The exact formulas of every loss in `models.losses`, and why the choice
between them matters for windows without a stream, are in
{doc}`datasets_and_models` ("Losses", "Formulas").

### Which of these does this project actually use?

Short answer, with the full reasoning in {doc}`datasets_and_models`: this
project's label went through two pivots (both logged in `PLAN.md`), and
which loss trains against which label matters a lot here, because it turned
out not to be a neutral choice:

1. An early binary/probability label (`rasterize.py`) paired with the
   segmentation losses above (Dice, Focal, Tversky, BCE) — still available,
   no longer the default.
2. `label_policy="stream_count"` (2026-09-09): a literal, per-channel star
   count, trained with `MSELoss`/`WeightedMSELoss`. This produced a model
   that localized streams well (high Dice *as an evaluation metric*, after
   thresholding the count prediction) but consistently under-recovered the
   true *amplitude* at the brightest pixels, by a large factor — a known
   failure mode of plain regression on a heavy-tailed target (most pixels
   near 0, a few very high), where minimizing average squared error
   rewards hedging toward the low end almost everywhere rather than
   committing to a rare, large value.
3. `label_policy="stream_detection"` (2026-09-15): since the actual goal
   is "is there a stream here," not "exactly how many stars," the count
   label is hard-thresholded into a binary target instead
   (`stream_raw > count_threshold`), and training goes back to the
   segmentation losses (`DiceLoss` paired with `head="sigmoid"` is this
   project's current default) — sidestepping the amplitude problem rather
   than fighting it, since a bounded target has no heavy tail to hedge
   against. The model's output is still a genuine continuous probability
   in `(0, 1)` at every pixel (not a boolean) regardless of this choice —
   see {doc}`datasets_and_models` for why the hard-threshold label doesn't
   make the *output* any less continuous.

`MSELoss`/`WeightedMSELoss` remain available for anyone who specifically
wants `label_policy="stream_count"`'s literal count back; the segmentation
metrics (Dice, IoU, precision/recall) are computed at *evaluation* time
either way, by thresholding whatever the model outputs, since they're a
useful, interpretable summary of "did the model find the stream" regardless
of which label trained it.
