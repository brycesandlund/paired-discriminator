# Class-matched CIFAR-10 comparison

Seed 0, vanilla/RSGAN/joint-paired, 10k and 50k generator-update checkpoints.
Each method trains once to 50k; both retained endpoints are evaluated afterward.
No checkpoint selection, method-specific tuning, or extra discriminator label
input is introduced. The original unconditional experiment remains unchanged.

## Changes from the unconditional experiment

The shared generator concatenates a 10-dimensional one-hot class label to the
128-dimensional Gaussian latent vector. Its first transposed convolution thus
has 138 input channels, adding 40,960 parameters identically for all methods.
Every other generator layer and the full discriminator architectures are
unchanged. All three conditional generators start with identical weights;
vanilla/RSGAN critics also start identically.

Each batch requests a near-balanced set of classes: each class occurs 12 or 13
times in the batch of 128. The extra classes rotate randomly, and labels are
shuffled. Each real example is sampled uniformly with replacement from the
requested class. The fake example uses that same requested label. D and G
updates use separate real, latent, and label streams, matched across methods.
Label streams are included in the resumable checkpoint, and the ordered training
labels have their own SHA-256 fingerprint.

- Vanilla: unary BCE on the class-balanced real and generated batches.
- RSGAN: BCE on `C(real_y) - C(G(z,y))`, with reversed targets for G.
- Paired: BCE on randomized A/B slots containing `real_y` and `G(z,y)`, with
  reversed targets for G. Both orientations occur equally often per batch.

All discriminators remain **unconditional**: they receive no label and have no
auxiliary class loss. In particular, this is not a standard label-conditioned
or projection-discriminator cGAN baseline. The generator is allowed to ignore
its label; class matching refers to the requested label, not a verified semantic
class of the generated image. A reference from the requested class makes the
pair closer only to the extent that the generator follows that request.

## Unchanged settings and evaluation

A10; float32 with TF32 disabled; batch 128; Adam 0.0002 and (0.5, 0.999); D:G 1:1;
no augmentation, batch normalization in D, spectral normalization, or gradient
penalty. Same BCE conventions, checkpoint frequency, and fixed evaluation noise
as the unconditional pilot. See [the base protocol](CIFAR10.md).

At 10k and 50k: 10,000 generated images, exactly 1,000 requested per class, versus
the original fixed 10,000-image real reference subset. FID, KID and feature-space
precision/recall retain the exact Inception feature extractor, preprocessing,
reference cache, random seeds, and settings. These are marginal distribution
metrics: good scores do not establish label adherence. Both endpoints remain
FID-10k evaluations despite their different training budgets.

Standard grids use 64 fixed latent draws and cycling requested classes. Additional
grids have one row per CIFAR class in canonical order: airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck. Every row reuses the same eight
latent vectors, so columns reveal the effect of changing the requested class.

Comparing against the old unconditional experiment changes conditioning,
initial generator dimensions/weights, and sampling as well as reference
selection. That comparison cannot isolate the causal effect of matching alone;
an unmatched-reference arm with the same conditional generator would be needed.
This is a one-seed experiment, not a claim of general superiority.

## Run and resume

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run modal_classmatched.py::prepare
uv run --extra cifar modal run modal_classmatched.py::verify_resume
uv run --extra cifar modal run --detach modal_classmatched.py::main --run-id cifar10-classmatched-v1 --steps 50000 --evaluate
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-classmatched-v1 results/
uv run --extra cifar python -m paired_discriminator.cifar_classmatched_report results/cifar10-classmatched-v1 results/cifar10-50k
```

Use an existing local destination for downloads. Persistent Volumes are shared
with the earlier pilot, but run directories and Modal app names are distinct.
Images are checked against the existing cached image order before storing aligned
labels. Source/config/label changes are rejected on resume. CPU and CUDA tests
check exact interrupted-versus-uninterrupted replay for all three methods.

## Why label use is not guaranteed here

There is a label-ignoring solution to the image-only objective. If G ignores y
and samples the overall real marginal p(x), the ordered real/fake pair has density
sum_y p(y) p(x_real|y) p(x_fake) = p(x_real) p(x_fake). Swapping its slots leaves
that density unchanged. Thus even the joint paired discriminator cannot penalize
label ignorance at this solution. Class matching at the sampler is not a guarantee
of semantic matching. The class grids are necessary to distinguish failure to
use the condition from failure of a genuinely same-class comparison.

The optional `modal_classmatched.py::score --run-id <id> --step 10000` entrypoint
can score saved 10k checkpoints while training continues. Completed metric files
are reused by the final run, preventing duplicate evaluations.
