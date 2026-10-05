# CIFAR-10 with conditional BatchNorm and projection discrimination

Seed 0; vanilla, RSGAN, joint paired; one trajectory to 50,000 updates per method,
retaining 10,000 and 50,000 checkpoints. This is a joint architectural intervention,
not an ablation separating the benefits of generator and discriminator changes.

## Generator

Noise is 128-dimensional standard normal. The initial one-hot concatenation is
removed. Three transposed-convolution hidden blocks (256,128,64 channels) each
use conditional BatchNorm and ReLU. BatchNorm has shared running statistics and
no built-in affine transform. A separate learned class table per block supplies
channelwise gamma and beta: gamma(y) * BN(h) + beta(y). Gamma starts at N(1,0.02),
beta at zero. All ten classes have their own parameters. The final RGB layer
and tanh are unchanged. G is identical across all three methods, including
initial weights. Initial weights are not claimed to match prior architectures.

## Discriminator

Three stride-2 convolution/LeakyReLU blocks retain widths 64,128,256. Inputs
contain RGB only (3 channels unary,6 paired), without broadcast label channels.
The final 4×4 feature map is sum-pooled into h of dimension 256. The old final
4×4 convolution is replaced by a linear scalar head and a 10×256 embedding table:

    score(x,y) = linear(h(x)) + dot(embedding(y),h(x))

For paired, h is a joint representation of the concatenated A/B images. Its score
still predicts A-real; it is not a separate realism score for each image. No
swap symmetry is enforced. Projection has multiplier 1, without normalization.
Convolution, linear, and projection embedding weights start at N(0,0.02), biases
at zero. Vanilla/RSGAN critics start identically. No spectral normalization,
auxiliary classifier, hinge loss, or other paper-specific architecture is added.
This tests the conditioning ideas, not a reproduction of a published full model.

Parameter counts: G 1,191,683 for all methods; D 661,697 unary and 664,769 paired.

## Controls and evaluation

The BCE objectives, reversed G targets for RSGAN/paired, sampling streams,
class-balanced batches, same-requested-class references, Adam settings, batch
128, D:G 1:1, deterministic float32, and checkpoint protocol are unchanged.
See [previous protocol](CIFAR10_CONDITIONAL.md) and [base protocol](CIFAR10.md).
G stays in train mode during both update passes, including running-stat updates.

All jobs request Modal A10; record actual hardware names in the run metadata.
All three methods see the same image counts, but have different decision counts
and compute costs. Keep both endpoint evaluations; no best-checkpoint selection.

Evaluation retains the identical 10,000 real references, 10,000 generated samples,
Inception-2048 FID/KID/precision/recall, and fixed noise. Evaluation requests exactly
1,000 generated examples per class. Same-latent class grids diagnose semantic
adherence visually. Pixel label sensitivity is not class accuracy. A standalone
validated classifier is not part of this run. One seed limits ranking claims.

## Run

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run modal_projection.py::verify_resume
uv run --extra cifar modal run --detach modal_projection.py::main --run-id cifar10-projection-v1 --steps 50000 --evaluate
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-projection-v1 results/
uv run --extra cifar python -m paired_discriminator.cifar_projection_report results/cifar10-projection-v1 results/cifar10-conditional-v1
```

Source/config/data/label/runtime guards prevent incompatible resume. Original
training modules and artifacts remain unchanged. Tests cover matching, gradients,
conditioning operations, initialization equality across methods, and exact replay.
