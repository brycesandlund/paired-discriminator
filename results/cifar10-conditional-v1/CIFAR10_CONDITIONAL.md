# CIFAR-10 with class-conditioned generators and discriminators

Seed 0; vanilla, RSGAN, and joint paired; one trajectory to 50,000 updates per
method, retaining and evaluating 10,000 and 50,000. No checkpoint selection.

This follows the G-only class-matched pilot, whose generators largely ignored
requested labels. Here every discriminator also receives the requested label.
A 10-dimensional one-hot vector is broadcast over the 32×32 image and concatenated
to its input channels. Unary critics receive 13 channels, joint paired receives
16 (two RGB images plus one shared condition). This adds 10,240 weights per
critic. All later layers, initialization rule, losses, optimizers, and sampling
remain unchanged. This is input concatenation, not a projection discriminator.
Changing the first layer's size changes discriminator initialization; the old
and new discriminator weights are not claimed to match. Vanilla and RSGAN do
start from identical critics within this experiment.

G is exactly the previous conditional generator, including initial weights:
128 Gaussian coordinates plus 10 one-hot coordinates. Real images are sampled
uniformly within requested class; each batch contains 12 or 13 examples per
class. Both paired inputs share the requested label. All three methods receive
the same numbers of real and fake images and matched sampling/noise streams.
Vanilla makes twice as many binary decisions as the paired methods; compute
cost and gradient scaling are not identical.

- Vanilla: BCE on D(real,y) and D(fake,y); G targets real.
- RSGAN: BCE on C(real,y) − C(fake,y); G reverses the target.
- Paired: BCE on D(A,B,y), predicting that A is real; G reverses the target.

Unlike G-only conditioning, the discriminator can detect a mismatch between the
requested label and image content. This supplies pressure to follow labels,
not a guarantee of successful optimization. Inspect same-latent class grids
at both endpoints before interpreting results as semantically matched pairs.

A10, float32, deterministic algorithms, TF32 disabled, batch 128, Adam 0.0002
with betas (0.5,0.999), 1 D update per G update. No augmentation, D batch
normalization, spectral normalization, gradient penalty, or auxiliary loss.
See [base protocol](CIFAR10.md) and [G-only protocol](CIFAR10_CLASSMATCHED.md).

Evaluation is unchanged: 10,000 generated images (1,000 requested per class),
same fixed 10,000 real training examples, Inception-2048 FID/KID/precision/recall.
These marginal metrics do not measure class accuracy. Fixed class grids have
10 rows in CIFAR order and eight identical latent draws across rows. Pixel
label sensitivity measures change, not semantic accuracy. One seed cannot
establish a reliable ranking, and this experiment does not isolate the benefit
of matching from conditioning relative to the unconditional pilot.

## Reproduce

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run modal_conditional.py::verify_resume
uv run --extra cifar modal run --detach modal_conditional.py::main --run-id cifar10-conditional-v1 --steps 50000 --evaluate
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-conditional-v1 results/
uv run --extra cifar python -m paired_discriminator.cifar_conditional_report results/cifar10-conditional-v1 results/cifar10-50k
```

Old training modules and results remain unchanged. Source, config, dataset,
labels, dependency manifests and runtime are guarded on resume. Checkpoints
include optimizers and every sampling RNG. CUDA verification compares 40
continuous updates against 20 plus 20 for every method.
