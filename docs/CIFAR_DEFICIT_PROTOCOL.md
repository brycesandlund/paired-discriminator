# CIFAR-10 D-only class deficit pilot

Prespecified 2026-10-06, before examining outcomes. Experiment #21 translates
#20's Gaussian-mode deficit sampler to classifier-estimated CIFAR class mass.

## Arms and workload

Fresh unconditional vanilla, paired, vanilla_deficit, and paired_deficit runs,
seed 0, 100,000 D/G updates each, evaluation every 10,000. These use the original
CIFAR DCGAN architecture and BCE objectives from `cifar.py`, width 64, latent
128, Adam learning rate 0.0002 and betas (0.5, 0.999). G receives no class labels;
D receives no class labels. No wider D, conditional BatchNorm, projection,
augmentation, or extra D updates.

Every D update consumes 128 real images and 128 generated images: 256 unary
classifications for vanilla or 128 real/fake pair classifications for paired.
Every G update uses a fresh batch of 128 generated images. Paired's 128 real G
references remain uniform, independent draws. One D and one G optimizer step
per iteration. Uniform arms are fresh controls in the current runtime; historical
CIFAR numbers are context, not substituted controls.

All four arms run the frozen ResNet56 on the existing D fake batch. Uniform arms
record the estimate but ignore it for sampling. Thus classifier monitoring
compute is shared; the adaptive sampler itself adds some overhead. No classifier
gradients, additional G samples, or changes to G/D loss functions.

## Sampling rule

Quantize each existing D fake image to uint8 as in retrospective evaluation,
then apply the established classifier normalization. Let a_k be the fraction of
**all 128 generated images** predicted as class k with softmax confidence >=0.9.
Rejected images remain in the denominator. This threshold is one of the prior
class-audit thresholds, fixed here before results; it is not a calibrated
probability that an image is valid.

Initialize q_k=0.1. Update q <- 0.99 q + 0.01 a after both optimizers.
The next iteration samples D's real class using:

- d_k = max(0.1 - q_k, 0)
- w_k = 0.05 + 0.5 d_k / sum(d), or uniform if sum(d)=0.

Draw classes with replacement, then images uniformly within each selected
training class. Each class keeps at least 5% probability. At low acceptance all
classes appear deficient, so weighting tends toward uniform. Only real-D RNG
consumption changes. EMA and all RNG/optimizer states are checkpointed.

The target for evaluation remains the original balanced CIFAR distribution.
The two adaptive arms estimate deficits from their own generators; they do not
share identical weight trajectories. This is supervised sampling using real
labels and a pretrained classifier, despite unconditional G/D architectures.

## Evaluation and interpretation

Each endpoint generates the same 10,000 fixed latent draws. Report:

- Raw class masses, class TV, and >=1%-mass coverage.
- Accepted mass, acceptance, augmented TV including rejected mass, and coverage
  at confidence thresholds 0.5, 0.7, 0.9, 0.95.
- Both the sampling classifier (ResNet56) and a separate evaluator (VGG16-BN),
  neither updated. VGG16-BN is not used to choose training samples; it remains
  correlated with ResNet56 through training data and task.
- FID, KID, precision, recall against all 10,000 official CIFAR test images;
  also the original fixed 10,000-image training reference for comparison.
- Fixed sample grids, first-in-draw-order galleries per predicted class,
  and the EMA/sampling-weight trajectories.

Classifier source revision and weight hashes are pinned to the prior audits.
Each classifier must exceed 90% real-test accuracy. No test images enter GAN
updates or the deficit EMA. Inception references reuse the integrity study's
hashed feature cache. Source/configuration/dataset/classifier hashes accompany
training; checkpoint, generated-array, and evaluation hashes accompany metrics.

A better class histogram alone is not evidence of better image fidelity or
within-class diversity. Confident misclassifications can bias sampling. This
single-seed pilot cannot establish robustness; the 3x3 failure in #20 motivates
subsequent replication if results justify it. No hyperparameter tuning is planned
within this pilot.

## Verification and execution

CPU tests check deficit mass and rejection, the sampling floor, class-index
sampling, detached classifier inference, exact resume, and bitwise equivalence
of uniform arms to original training. A real A10 preflight verifies exact resume
for all four arms, original uniform trajectories, and unchanged G/noise RNGs
before production jobs are spawned.

```sh
uv run --extra cifar pytest tests/test_cifar_deficit.py -q
uv run --extra cifar modal run --detach modal_deficit.py --run-id cifar10-deficit-v1 --steps 100000
```

Four A10 jobs each advance through training/evaluation segments. Progress and
results persist in Modal volume `paired-discriminator-cifar10-runs`, under
`cifar10-deficit-v1`. Each arm has an exact-resume checkpoint every 1k and retained
G checkpoints every 10k. Restarting uses the same command and refuses changed
training provenance. Local results live in `results/cifar10-deficit-v1`.
