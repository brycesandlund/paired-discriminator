# Native-image flow-matching pilots (#35)

User requested flow matching on the MNIST protocol and CIFAR in parallel.
Start with 10k updates, five MNIST seeds0–4 and CIFAR seed0, matching the original
GAN 10k endpoints. No automatic 50k extension. This short pilot establishes
sampling quality and cost before committing to longer image runs.

## Training

Unconditional independent-pair straight-line flow matching. Draw real x1
uniformly with replacement from the full training split, standard-normal image
noise x0 independently, and t uniformly on [0,1]. Train velocity v(x_t,t) on
MSE to x1-x0, with x_t=(1-t)x0+t*x1. Pixels normalized to [-1,1]. No labels,
classifier loss, deficit sampling, optimal-transport pairing, augmentation, or
extra reconstruction objective. This is the ordinary velocity regression
objective, not an auxiliary loss added to a GAN.

Native MNIST 1×28×28 and CIFAR10 3×32×32. Same respective 60k/50k training images
as the GAN experiments; official test images excluded from updates. Flow uses
full-image Gaussian noise rather than the GAN's 128-dimensional noise vector.

Small convolutional U-Net: widths32/64/128, two stride2 downsamplings, residual
GroupNorm/SiLU blocks, nearest-neighbor upsampling and skip concatenations.
Sinusoidal time embedding injected into each block. No attention, pretrained
features, class conditioning, dropout, or output tanh. This is a new baseline
architecture, not a parameter-matched GAN. G-like decoder architectures alone
cannot represent an image-valued velocity conditioned on a noisy input image.

Batch128; Adam lr .0002, betas(.9,.999), linear warmup500 updates then constant.
Float32, deterministic CUDA algorithms, TF32 off. Raw weights fixed as primary;
EMA(.999) retained in checkpoints but not used to select reported results.
One flow optimization step consumes128 real images and one noise/time batch.
At 10k that is1.28M real draws, matching GAN D real draws but excluding the
paired GAN's additional G references. Updates and draw counts do not equate
compute. All three sampling streams are explicit, independent and checkpointed.

## Evaluation

Full 10k-sample evaluation at10k training updates. This deliberately differs
from the GAN MNIST every1k full evaluation because ODE sampling is more costly.
Every1k retain a resumable checkpoint and fixed64-sample midpoint16 preview.
Previews are not the final metric sample set.

Primary solver: explicit midpoint64 steps (128 network evaluations/image).
On seed0 for each dataset, repeat the full evaluation with midpoint128 (256
network evaluations/image), same noise, to expose solver sensitivity. A fixed
CPU seed99174 generates image-shaped noise; latent dimensions differ from GANs,
so equal RNG seed does not imply matched generated images. Integrate without
clipping. At the endpoint, round/clamp to uint8 exactly as the GAN evaluation
does. Report pre-clipping out-of-range pixel fraction. A saved-model replay
must reproduce the first float batch exactly.

MNIST: original frozen classifier, checkpoint hash and empirical training digit
frequencies. Same raw/accepted digit mass, 1%-mass coverage, confidence threshold
.9 and augmented TV (rejected mass retained). Same first-ten-per-predicted-digit
galleries; no confidence selection. Compare with original GAN seed-matched10k
metrics; report mean and sample SD across seeds. No within-digit diversity claim.

CIFAR: same frozen ResNet56/VGG16-BN weights, class TV, coverage, confidence
thresholds, and evaluator-agreement diagnostic. Reuse Inception2048 features
and FID/KID/precision/recall implementation from the integrity/deficit studies.
Report official-test held-out references and the original fixed train10k
reference separately. Do not compare a held-out score to an old train-reference
score without labeling it. One seed is a pilot, not a reliable method ranking.

## Verification and operation

Local tests check path endpoints/target, time-dependent integration, native
shapes/time gradients, and exact resume/evaluation neutrality. A CUDA preflight
checks20 versus10+10 updates for both datasets with optimizer, raw/EMA weights,
loss logs and explicit RNG states identical. Data/config/source/runtime hashes
are enforced at resume. Earlier experiment artifacts remain unchanged.

`uv run --extra cifar modal run --detach modal_image_flow.py::main --run-id image-flow-v1 --steps 10000`

Six concurrent A10 jobs at most, persistent existing dataset volumes and a new
`paired-discriminator-image-flow-runs` result volume. Training checkpoints every
1k, bounded jobs, one retry. Results/plots/source/metrics collected under
`results/image-flow-v1`; large weights and arrays remain in the volume and are
ignored by Git. No commits. Record training and ODE inference time separately.

The independent linear-interpolation velocity objective follows the general
[flow-matching formulation](https://arxiv.org/abs/2210.02747); this implementation
is a small pilot, not a reproduction of a published image benchmark.
