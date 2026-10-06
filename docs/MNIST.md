# Ordinary MNIST: unconditional vanilla versus paired

This experiment tests digit representation directly, following the eight-Gaussian
ring results. It uses ordinary MNIST, not stacked MNIST. No CIFAR training is
performed. The comparison is fixed before looking at GAN results; it does not
weaken the baseline or select a collapse-prone seed afterward.

## Comparison

- Vanilla versus paired, seeds 0–4, trained from initialization to 50k updates.
- Evaluate every 1k on 10,000 fixed generated images; highlight 10k and 50k.
- Native 28×28 grayscale images. Use all 60,000 training images, uniformly sampled
  with replacement, normalized to [-1,1]. Official test images are excluded from
  GAN training. Training is unconditional: digit labels never enter G, D, or
  reference selection. Digit proportions follow the empirical training set.
- Every D update sees 128 real and 128 fake images. Vanilla produces 256 unary
  scores; paired produces 128 joint scores on randomized two-channel inputs.
- Every G update uses 128 fresh generated images. Paired additionally uses 128
  fresh independent real references. Both perform one D and one G optimizer
  update per iteration. Equal sample counts do not imply equal FLOPs.
- BCE objectives as in previous experiments: vanilla non-saturating G; paired D
  predicts which slot is real and G reverses that label. Exactly 64 pairs have
  real in slot A and 64 in slot B in each paired batch.

## Architecture and optimization

Both methods share identical G initial weights within a seed. A 128-dimensional
standard-normal latent vector feeds a linear layer to 128×7×7; BatchNorm/ReLU;
4×4 stride-2 transposed convolution to 64×14×14; BatchNorm/ReLU; 4×4 stride-2
transposed convolution to 1×28×28 with tanh. G has 941,569 parameters.

D uses three 4×4 stride-2 padding-1 convolution blocks, channels 64→128→256,
spatial sizes 14→7→3, each with LeakyReLU(0.2). A linear layer on flattened
256×3×3 features produces one logit. Vanilla input has one channel, paired input
two. No D BatchNorm, label conditioning, or cross-batch comparison layer.
D has 659,137 parameters for vanilla and 660,161 for paired. The paired D joins
both inputs immediately; no antisymmetry or separate encoders are imposed.

Adam 2e-4, betas (0.5,0.999), no learning-rate schedule, augmentation, spectral
normalization, gradient penalty, clipping, or EMA. Conv/linear weights N(0,0.02),
bias zero; BatchNorm scale N(1,0.02), offset zero. Float32, deterministic CUDA
algorithms, TF32 off. G is in training mode during the no-gradient D fake pass
and during its own update, as in our previous runs.

Training RNG streams for real draws, latent draws, and slot permutations are
explicit and saved. Evaluation uses an independent CPU RNG and G.eval(), then
restores G's mode. Exact resume and evaluation neutrality are checked on CUDA.
Snapshots retain G and D every 10k; full optimizer/RNG checkpoints every 1k.
A checkpoint whose evaluation was interrupted is evaluated on resume before the
next training update.

## Frozen evaluation classifier

A separate CNN maps each image to ten digit probabilities. It uses 3×3
convolutions (32,64), ReLU and 2×2 pooling, then a 128-unit hidden layer with
ReLU/dropout(0.25) and ten logits. Inputs use MNIST normalization
(x−0.1307)/0.3081. Train with Adam 0.001, batch128, for ten epochs on a fixed
50,000-image subset of MNIST training data. Select highest accuracy on the
remaining 10,000 training images, breaking ties in favor of the earliest epoch.
Only then report official-test accuracy and confusion matrix. Test data never
selects epochs. The classifier is frozen and never supplies GAN gradients.
Its checkpoint and source hashes are included in every run and metric record.

Probabilities are **uncalibrated**. Confidence is a diagnostic, not a guarantee
that a generated image is a recognizable digit. The classifier may confidently
assign a class to malformed or out-of-distribution images.

## Metrics

Let q_i be the fraction of the full generated evaluation batch classified as
digit i, and p_i be that digit's empirical MNIST training-label frequency.

- **Digit coverage:** number of digits with q_i ≥ 0.01 (at least 100 of 10,000
  generated images). Also report the number observed at least once.
- **Mode TV:** ½ Σ_i |q_i − p_i|. Lower means better digit-frequency agreement.
  It measures between-digit balance, not handwriting-style diversity or fidelity.
- **Confidence-filtered coverage:** same 1% threshold, but count only images with
  maximum classifier probability ≥0.9. Denominator remains all generated images.
- **Accepted mass per digit:** accepted images of each predicted digit divided by
  all generated images. Do not renormalize away uncertain samples.
- **Confidence-augmented TV:** ½(Σ_i |a_i−p_i| + rejected_fraction), where a_i
  denotes accepted mass. The ideal target has zero rejected mass. Even real images
  need not score zero; report the official-test diagnostic as a reference. This
  score is a classifier-based proxy, not a calibrated image-validity measure.
- Confidence histogram and quantiles, plus fixed-noise random sample grids.

At 10k and 50k, show the first ten examples in fixed random evaluation order for
each predicted digit. There is no sorting by confidence or visual selection.
Empty cells indicate fewer than ten examples were available. These galleries
help identify confidently misclassified images and repeated handwriting styles.
Ten categories are a coarse partition, not a claim of exactly ten density modes.

All summaries show individual seeds and mean ± sample standard deviation.
Reference sampling noise is estimated by 1,000 independent 10,000-label draws
from the empirical training distribution. The same evaluation noise is reused
across checkpoints/methods, so checkpoint results are correlated.

## Running and storage

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run --detach modal_mnist.py::main --run-id mnist-v1 --steps 50000
uv run --extra cifar modal volume get paired-discriminator-mnist-runs /mnist-v1 results/
uv run --extra cifar python -m paired_discriminator.mnist_report
```

The existing `cifar` dependency extra supplies torch/torchvision/Modal; it does
not imply CIFAR data or training. Data/classifier live in Modal volume
`paired-discriminator-mnist-data`; GAN artifacts in
`paired-discriminator-mnist-runs`. Large tensors, raw generated images, and
per-image probability arrays are excluded from Git, but retained locally/Modal.
Plots, source snapshots, configuration, and JSON/CSV summaries are kept in Git.

## Completed run

All ten trajectories reached 50k, with 500 evaluations. The frozen classifier
selected epoch 8 and scored 99.14% on official test images. All GAN runs requested
A10; Modal assigned NVIDIA A10G to vanilla seeds 2 and 4 and paired seed 1, and NVIDIA
A10 to the others. Treat recorded runtimes as observed costs, not a strictly
hardware-controlled benchmark. All 63 local tests and CUDA verification passed.

[Results](../results/mnist-v1/REPORT.md) include per-seed curves, digit mass and
confidence-filtered mass, classifier checks, and image galleries.
