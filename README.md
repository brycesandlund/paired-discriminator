# paired-discriminator

Does jointly comparing a real and generated sample improve mode coverage over a
vanilla GAN? The first experiment compares only those two methods on an eight-mode
2-D Gaussian ring. No reference reconstruction loss or PairGAN/RelationGAN loss is used.

For the running experiment history, results, and open questions, see
[training_runs.md](training_runs.md).

## Setup

```sh
uv sync --python 3.12
uv run pytest
```

Dependencies live in the project's `.venv`; `uv.lock` records exact versions.

## Run

```sh
# Smoke check: two methods, one seed, 50 updates each.
uv run paired-discriminator --output results/smoke --steps 50 --seeds 0

# Prespecified comparison: two methods × five paired seeds × 10,000 updates.
uv run paired-discriminator --output results/ring8-v1

# Deterministic extension to 50,000 updates, preserving the original run.
uv run paired-discriminator --config configs/ring8-50k.json --output results/ring8-50k
uv run python -m paired_discriminator.compare results/ring8-v1 results/ring8-50k
```

The output directory must not already exist. To change an experiment, copy
`configs/ring8.json` and pass `--config path/to/config.json`. CPU with one thread
is the default for these small networks. `--device mps` and `--device cuda` are
available, but support and throughput should be checked before a sweep.

## Objectives

Vanilla D uses BCE with real=1 and generated=0. Its loss is averaged over both
halves of the batch. G uses non-saturating BCE with generated=1.

Paired D receives a concatenated real/generated point pair with a random,
balanced assignment to slots A/B. D uses BCE to predict whether A is real.
G uses BCE with that label reversed; gradients flow through the generated slot,
with D's parameters frozen. D is an unrestricted joint MLP, not a difference
of independently computed scores. It may learn to ignore a slot; this experiment
does not artificially prevent that shortcut.

Both methods use the same 16-dimensional noise, generator (two 128-wide hidden
layers), discriminator hidden widths, LeakyReLU activations, Adam settings,
batch 256, and 1:1 D/G update ratio. D parameters differ by 256 weights because
the paired input has four coordinates instead of two. Random streams for data,
noise, slot assignment, initialization, and evaluation are separated. Generator
initial weights and training data/noise streams match within each seed pair.

## Evaluation

Eight Gaussians have equal weights, radius 2, and per-axis sigma 0.1. Evaluation
uses 10,000 fixed latent samples every 1,000 updates and at initialization.
No best-checkpoint selection or early stopping is used.

- Valid: distance to nearest mode center is at most 3 sigma (0.3).
- Covered: at least 1% of all generated evaluation samples are valid in that mode.
- Mode mass: accepted count in each mode divided by the total generated count.
- Mode TV: total variation from eight target weights of 1/8 and an extra invalid
  category with target weight zero. Lower is better; off-mode mass is penalized.

The target itself has about 98.9% mass inside a 3-sigma radius in two dimensions.
Good coverage alone is not sufficient: validity and mode balance also matter.
These metrics do not fully assess the within-mode distribution shape.

## Artifacts

Each suite saves configuration, a source snapshot with SHA-256 hashes and lockfile,
per-run CSV metrics, final samples, final model/optimizer/RNG checkpoints, machine
metadata, and a report with all-seed plots. Reports, plots, metrics, source snapshots,
and small final-sample arrays are intended for Git. Model/optimizer checkpoints,
console logs, smoke checks, reproducibility checks, and `results/scratch/` are
git-ignored. Checkpoints are saved locally for inspection; automatic resume is
not implemented.

The first completed comparison is in [results/ring8-v1/REPORT.md](results/ring8-v1/REPORT.md).
The longer-run comparison is in [results/ring8-50k/EXTENSION.md](results/ring8-50k/EXTENSION.md).
It verifies that all original evaluation rows replay exactly, independently
recomputes final metrics from saved samples, and summarizes the final 10,000
updates as well as the endpoint. The extension changes only the update budget.
Regenerate its plots and report from the saved metrics and sample arrays with:

```sh
uv run python -m paired_discriminator.report results/ring8-v1
```

This first comparison is an untuned pilot, not a claim of general superiority.
Equal steps are not equal FLOPs; measured training time is included. The same
hyperparameters may favor one method. Follow-up tuning should give both methods
equal budgets, and a relativistic baseline can help isolate joint processing.

## Relativistic baseline

RSGAN uses the vanilla unary critic architecture with logit `C(real) - C(fake)`.
D uses mean BCE with target 1; G uses target 0 (non-saturating label reversal).
This follows [the author's RSGAN implementation](https://github.com/AlexiaJM/RelativisticGAN).
It is distinct from the batch-average RaSGAN variant. Initial critic weights match
vanilla, and initial G weights and sampling streams match all three methods.

```sh
uv run paired-discriminator --config configs/ring8-50k.json --methods rsgan --output results/ring8-rsgan-50k
uv run python -m paired_discriminator.relativistic_report results/ring8-relativistic-comparison
```

The comparison reuses existing vanilla/paired runs and reports 10k and 50k budgets
on the same trajectories. RSGAN saves the 10k samples during the longer run.
The assembly independently checks all 30 endpoint sample sets against recorded
metrics. See [comparison report](results/ring8-relativistic-comparison/REPORT.md).

## CIFAR-10 on Modal

The three-method image pilot uses a shared convolutional generator and matched
training settings, with an A10 job per method. See [the protocol and commands](docs/CIFAR10.md)
for architecture, losses, fixed evaluation, checkpoint/resume, and storage.
CIFAR/Modal dependencies are optional: `uv sync --extra cifar`.

The seed-0 CIFAR extension is in
[results/cifar10-50k/EXTENSION.md](results/cifar10-50k/EXTENSION.md), comparing
unchanged 10k artifacts with resumed 50k checkpoints under the same evaluation.

## Class-matched CIFAR-10

The follow-up gives all three generators a class label, with same-requested-class
real references for RSGAN and paired. Discriminators still receive images only.
See [the class-matched protocol](docs/CIFAR10_CLASSMATCHED.md) for controls,
matching checks, class-organized grids, and the 10k/50k run command.

The completed seed-0 class-matched comparison, including both budgets and
requested-class grids, is in
[results/cifar10-classmatched-v1/REPORT.md](results/cifar10-classmatched-v1/REPORT.md).

## CIFAR-10 with labels in G and D

Follow-up to the G-only class-matched pilot: all three discriminators receive
broadcast one-hot class labels. See [protocol](docs/CIFAR10_CONDITIONAL.md).
Run with `modal_conditional.py::main`; historical experiments remain separate.

Completed seed-0 results: [report and grids](results/cifar10-conditional-v1/REPORT.md).

## Conditional BatchNorm and projection

Experiment #8 applies conditional BatchNorm in G and projection in every D.
See [protocol](docs/CIFAR10_PROJECTION.md),
[results and class grids](results/cifar10-projection-v1/REPORT.md), and
[the running overview](training_runs.md).

## Integrity study through 100k

Experiment #9 retains G and D every 10k and evaluates held-out distribution
metrics, discriminator train/held-out behavior, and nearest neighbors.
See [protocol](docs/CIFAR10_INTEGRITY.md), [full report](results/cifar10-integrity-v1/REPORT.md),
and [training overview](training_runs.md).

### Discriminator batch and capacity probes

Two paired-only arms keep G unchanged while independently increasing D batch
to 256 pairs or D width to 128. See the [protocol](docs/CIFAR10_CAPACITY.md)
and [100k comparison report](results/cifar10-capacity-v1/REPORT.md).

### Ordinary MNIST

Unconditional vanilla versus paired over five seeds, tracking digit coverage and
frequency balance through 50k updates. See the [protocol](docs/MNIST.md)
and [five-seed results](results/mnist-v1/REPORT.md).

### Retrospective CIFAR class representation

The original unconditional checkpoints now have a [class-balance audit](results/cifar10-class-representation-v1/REPORT.md) using two validated frozen classifiers. Paired has lower raw class TV at 50k under both evaluators, despite worse FID; generated-image ambiguity and the single seed limit the conclusion.

The [conditional CBN/projection audit](results/cifar10-conditional-representation-v1/REPORT.md) evaluates the original three methods every 10k through 100k. Paired trails both baselines on requested-class adherence under both classifiers at every checkpoint; class balance shows no consistent paired advantage.

### Ring mode-count sweep

The [8/16/32-component comparison](results/ring-mode-count-v1/REPORT.md) holds radius and Gaussian width fixed. It includes all three methods and five seeds, plus fine spatial metrics and a smooth-ring control to detect misleading coverage as components get closer.

### Fixed-spacing Gaussian grids

The [3×3 / 5×5 / 7×7 grid experiment](results/grid-mode-count-v1/REPORT.md) expands outward at fixed spacing 1.5 and sigma 0.1, with five seeds per method. Paired wins mode and fine-grid TV at 50k in every matched seed for 9/25 modes; the advantage is not present at 49 modes within this budget.

The [7×7 continuation to 150k](results/grid7-extension-comparison-v1/REPORT.md) reverses the 50k ranking: paired beats both baselines on mode TV in all five seeds at 100k and 150k, and on fine-grid density TV in all five at 150k. Recovery remains incomplete.

The [PacGAN2 comparison](results/grid-pacgan2-comparison-v1/REPORT.md) adds packed
real-real versus fake-fake discrimination with the same D architecture as paired.
Five seeds per grid, 50k updates on 3×3/5×5 and 150k on 7×7; existing baselines
are reused. D consumes 256 real and 256 fake points; G trains on 256 fresh fake
points grouped into 128 packs, with gradients through both members.

```sh
uv run python -m paired_discriminator.grid_pacgan_run
uv run python -m paired_discriminator.grid_pacgan_report results/grid-pacgan2-comparison-v1
```

The [two-output discriminator comparison](results/grid-dual-slot-comparison-v1/REPORT.md)
trains two logits to classify each slot independently using balanced RR/RF/FR/FF
pairs. G trains on FF/FR/RF pairs, applying losses only to generated slots.
It retains 256 real and 256 fake points per D update and 256 fresh generated
points per G update, with the same grid budgets and five seeds.

```sh
uv run python -m paired_discriminator.grid_dual_slot_run
uv run python -m paired_discriminator.grid_dual_slot_report results/grid-dual-slot-comparison-v1
```

### D-only batch doubling on grids

The [D512 comparison](results/grid-d512-comparison-v1/REPORT.md) doubles D's
real and generated sample batches for paired, PacGAN2 and two-output while
keeping G's batch at 256. It reports both equal G updates and equal cumulative
D samples, using extra 5k/25k/75k checkpoints for the latter. Five seeds per
method/grid; 3×3/5×5 through 50k and 7×7 through 150k.

```sh
uv run python -m paired_discriminator.grid_d_batch_run
uv run python -m paired_discriminator.grid_d_batch_report results/grid-d512-comparison-v1
```

### Grid reference selection

The [D-only reference experiment](results/grid-reference-comparison-v1/REPORT.md)
compares nearby one-to-one pairing with adaptive sampling of underrepresented
Gaussian modes, including vanilla with the same reweighting rule. G references
stay random. The [protocol](docs/GRID_REFERENCE_PROTOCOL.md) specifies the
matching, EMA, probability floor and real-versus-real diagnostic.

```sh
uv run python -m paired_discriminator.grid_reference_audit
uv run python -m paired_discriminator.grid_reference_run
uv run python -m paired_discriminator.grid_reference_report results/grid-reference-comparison-v1
```

### CIFAR class deficit sampling

The [CIFAR deficit protocol](docs/CIFAR_DEFICIT_PROTOCOL.md) transfers the grid
sampler to classifier-estimated class deficits. Four unconditional arms compare
vanilla/paired with uniform/adaptive D real sampling, keeping G unchanged.
ResNet56 guides sampling; VGG16-BN supplies a separate class audit. Held-out
FID/KID/precision/recall and both classifiers are evaluated every 10k through
100k, seed 0. See experiment #21 in [training_runs.md](training_runs.md).

```sh
uv run --extra cifar modal run --detach modal_deficit.py --run-id cifar10-deficit-v1 --steps 100000
uv run --extra cifar python -m paired_discriminator.cifar_deficit_report --sync
```

### G-only grid deficit references

[Protocol](docs/GRID_REFERENCE_G_PROTOCOL.md): original uniform D training,
deficit-weighted real references only during paired G updates. Five seeds
on each grid; compares against frozen uniform and D-only deficit runs.

```sh
uv run python -m paired_discriminator.grid_reference_g_run
```
