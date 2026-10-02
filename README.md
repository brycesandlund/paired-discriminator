# paired-discriminator

Does jointly comparing a real and generated sample improve mode coverage over a
vanilla GAN? The first experiment compares only those two methods on an eight-mode
2-D Gaussian ring. No reference reconstruction loss or PairGAN/RelationGAN loss is used.

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
