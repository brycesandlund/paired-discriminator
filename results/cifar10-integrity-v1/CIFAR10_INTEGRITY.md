# CIFAR-10 integrity study

Prespecified scope: CBN/projection architecture from experiment #8; vanilla,
RSGAN, paired; seed 0; 100,000 updates; evaluation every 10,000 updates. No
hyperparameter changes, checkpoint selection, or adaptive early stopping.
This is a diagnostic study of one seed, not a replicated method ranking.

## Training and retained artifacts

Replay from initialization to recover missing 20k/30k/40k models, then continue
to 100k. Preserve both G and D weights every 10k; retain the latest full
optimizer/RNG state every 1k. The new training module changes only checkpoint
retention, not architecture, sampling, losses, or optimization. Compare replayed
10k/50k G weights and 50k D weights against experiment #8 where available.

[Architecture and optimizer protocol](CIFAR10_PROJECTION.md).
Run ID: `cifar10-integrity-v1`; separate from historical runs.

## References and generator evaluation

Train on all 50,000 official CIFAR-10 training images. The 10,000 official test
images are never supplied to a GAN update. Use them as held-out diagnostic
references, not as input to G. Because we inspect these results, they should not
later be portrayed as an untouched final test for decisions based on this study.

At every checkpoint, generate the same 10,000 fixed-noise images (1,000 requested
per class). Compute FID, KID, precision, and recall against both:

- The previous fixed random 10,000-image training reference, preserving historical comparability.
- All 10,000 held-out images.

Use the same torch-fidelity 0.4.0 Inception-2048 features, uint8 conversion,
KID 100 subsets of 1,000, and precision/recall k=3 as before. Retain the full
training and held-out feature caches for reproducibility. Report a real-training
10k versus real-held-out 10k baseline to contextualize finite-reference differences.
Held-out FID/KID do not detect memorization reliably by themselves. Metrics pool
classes; they do not measure semantic class adherence.

## Discriminator train/held-out comparison

Construct a fixed class-balanced training pool (1,000 per class) and reorder the
held-out pool to the identical cycling label sequence. Use all 10,000 generated
floating-point images before evaluation quantization, with exactly the same fake
image and label for each train/held-out comparison. D runs in eval mode.

- Vanilla: compare real-image logits, real-target BCE, and real acceptance. Report shared fake rejection/BCE separately.
- RSGAN: compare margins C(real,y)-C(fake,y), BCE target 1, and sign accuracy.
- Paired: evaluate BOTH A/B orientations. Negate the B-real logit to align signs,
  then average margins; separately average the two BCE losses and accuracies.

A higher aligned margin always means stronger acceptance of the real input.
Report train and held-out means, quantiles, per-class means, train-minus-held-out
margin, and train-membership AUC using margin as the score. AUC 0.5 means no
ranking preference; values above 0.5 mean training examples tend to score higher.
This is a diagnostic, not a calibrated membership attack or a proof of causality.
Absolute margins are not directly comparable between methods or checkpoints.
Preserve per-example arrays for later analysis.

## Nearest neighbors and copying checks

For every generated image, find nearest neighbors in:

1. A fixed balanced 10,000-image training pool.
2. The 10,000 held-out images (equal candidate counts for the comparison).
3. All 50,000 training images (more complete search for potential copying).

Search both raw Inception-2048 Euclidean distance and RGB pixel RMS distance on
[0,1]. Use chunked exhaustive distance calculations and recompute selected
neighbor distances by direct subtraction to reduce cancellation error. Record
indices and distances, summaries, and fraction closer to training in equal pools.
The 50k pool naturally offers closer neighbors; do not interpret its distance
advantage over a 10k pool as overfitting.

Compute a held-out-real-to-training10k distance baseline in both spaces. Save
triplets (generated, closest training50k, closest held-out) for eight fixed queries
and eight queries closest to training in each space. These latter queries are
explicitly selected to inspect suspiciously close examples, not a representative
quality sample. Check exact uint8 matches by image hashes and generated duplicates.
Audit exact train/test duplicates in the source dataset. Similarity is evidence
to inspect, not proof of copying; no exact matches does not rule out memorization.

## Validation and execution

Tests cover unchanged training trajectories, resume, label/order matching,
known discriminator gaps, slot orientation, and nearest neighbors against a
brute-force reference. CUDA verifies interrupted replay for every method.
Reference arrays and source are fingerprinted. Evaluations reject reuse after
source changes rather than silently mixing protocols.

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run modal_integrity.py::verify_resume
uv run --extra cifar modal run modal_integrity.py::prepare_integrity
uv run --extra cifar modal run --detach modal_integrity.py::main --run-id cifar10-integrity-v1 --steps 100000
uv run --extra cifar modal run --detach modal_integrity.py::score_range --run-id cifar10-integrity-v1
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-integrity-v1 results/
uv run --extra cifar python -m paired_discriminator.integrity_report results/cifar10-integrity-v1
```

The evaluator reads retained checkpoints and never changes training models or RNGs.
