# CIFAR-10 pilot protocol

This moves the eight-mode experiment to unconditional 32×32 RGB generation. The
first run is one matched seed (0) for vanilla, RSGAN, and the joint paired
discriminator, evaluated at exactly 10,000 generator updates. A later extension
can resume the same trajectories to 50,000 updates; a five-seed sweep is separate
from this initial implementation/feasibility pilot.

## Model and training

`configs/cifar10.json` fixes the settings before inspecting pilot outcomes:

- G: 128-dimensional normal noise, transposed convolutions producing 4×4×256,
  8×8×128, 16×16×64, and 32×32×3, with batch normalization/ReLU in hidden layers
  and tanh output. DCGAN-style initialization.
- D: stride-two convolutions at widths 64, 128, 256, then a scalar convolutional
  head. LeakyReLU(0.2), no batch normalization, dropout, spectral normalization,
  gradient penalty, augmentation, or other regularization. In particular, the
  unary critics cannot obtain batch comparisons through batch normalization.
- Vanilla and RSGAN have three input channels and identical initial D weights;
  paired has six input channels. All methods have identical initial G weights.
- Batch 128 real + 128 fake per D update, one D update per G update, Adam at
  0.0002 with betas (0.5, 0.999), float32 with TF32 disabled.
- Independent uniform sampling with replacement from the 50,000 training images;
  no labels enter training. Pixels map from uint8 to [-1, 1].
- Dedicated matched random streams for D/G real images, D/G latent draws, and
  balanced randomized pair slots. Fresh real/noise draws for the G update.
- G remains in train mode in both updates, including its batch-normalization
  running-stat updates; fixed-grid generation temporarily uses eval mode.

Vanilla D averages BCE across 256 unary decisions. RSGAN averages BCE on
`C(real) - C(fake)`, target 1 for D and 0 for G, following the
[author's implementation](https://github.com/AlexiaJM/RelativisticGAN).
Paired averages BCE across 128 randomized-slot joint decisions, with reversed
labels for G. D is frozen during G optimization; G is detached during D
optimization. Equal sample/update budgets are not equal FLOPs or gradient scales.

## Evaluation

Every 1,000 steps: a fixed 64-image grid and resumable checkpoint. At 10k and 50k:
a retained generator checkpoint for independent evaluation.

Pilot evaluation uses 10,000 generated images and a fixed uniformly selected
10,000-image subset of CIFAR-10 train, with the same evaluation seeds across
methods. This is **FID-10k**, not a standard 50k-generated-image FID benchmark.
No image selection, best-checkpoint selection, or early stopping is used.

`torch-fidelity` computes FID, KID (100 subsets of 1,000), and improved precision
and recall (k=3). All use the explicitly selected `inception-v3-compat` 2048-D
features, including precision/recall (not its possible default VGG embedding).
Generated values map from [-1, 1] to rounded/clamped uint8 pixels before library
preprocessing. Reference features are cached in the persistent data Volume.

Precision and recall provide complementary fidelity/diversity diagnostics; they
are not literal CIFAR class coverage and do not establish absence of collapse.
KID subset standard deviation is not between-training-seed uncertainty. The
one-seed pilot cannot establish method superiority. CIFAR labels are not used for
metrics in this pilot.

## Modal operation

Install the optional project dependencies locally:

```sh
uv sync --extra cifar
uv run --extra cifar pytest -q
# Authenticate with `uv run --extra cifar modal setup` if needed.
uv run --extra cifar modal run --detach modal_app.py --run-id cifar10-smoke-v1 --steps 300
uv run --extra cifar modal run modal_app.py::verify_resume
uv run --extra cifar modal run --detach modal_app.py --run-id cifar10-pilot-v1 --steps 10000 --evaluate
```

The Modal image installs from `uv.lock`; local source is uploaded without
rebuilding dependency layers. Training uses one A10 per method, at most three
concurrent training containers. Evaluation is sequential, using the same GPU
class. The code does not deploy a continuously running service.

Persistent Volumes are `paired-discriminator-cifar10-data` (dataset and feature
cache) and `paired-discriminator-cifar10-runs` (one subdirectory per run/method/seed).
Each job owns its files. The latest checkpoint is atomically replaced and explicitly
committed every 1,000 steps. It includes G/D, both optimizers, all sampling streams,
global RNG state, training step, losses, and cumulative training time. Retries
resume from this checkpoint. Source/config/dataset/runtime and lockfile changes
are rejected for existing runs; use a new run ID for a changed experiment.

Both functions have bounded timeouts; training has one retry. A 50k extension can
use the same run ID and `--steps 50000 --evaluate` after runtime is assessed.
Checkpointing does not itself guarantee cross-device/version bitwise identity.
CPU and A10 resume tests explicitly check the supported execution paths.

Download results with:

```sh
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-pilot-v1 results/
```

Training checkpoints are ignored by Git; small metrics, logs in CSV, source
snapshots, dependency manifests, and sample grids can be committed. No automatic
Git staging or commits are performed.
