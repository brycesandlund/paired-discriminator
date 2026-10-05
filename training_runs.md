# Training runs

Last updated: 2026-10-05. Running overview of completed research experiments.
Detailed reports, configurations, and raw metrics are linked below. Add future
experiments here, preserving earlier results and distinguishing new trials from
extensions or reused checkpoints.

## Where we stand

**Paired improved mode coverage and balance on the eight-Gaussian toy task across
five seeds. That advantage has not carried over to our CIFAR-10 pilots.** On
CIFAR, paired starts behind and improves with longer training, but has not shown
a consistent advantage over vanilla or RSGAN. All CIFAR comparisons have only
one training seed and one untuned hyperparameter setting.

Our reference-proximity explanation is still a hypothesis. The G-only
class-matched experiment largely failed to induce label use. Giving labels to
both G and D increased label sensitivity, but semantic class adherence remained
incomplete and paired's marginal metrics worsened. Conditional BatchNorm and
projection conditioning (#8) subsequently improved FID/KID for all three methods,
but paired still trailed the baselines at 50k and class adherence remained incomplete.

The integrity extension (#9) finds a growing discriminator train/held-out gap,
while generator distribution metrics improve more slowly and train/held-out FIDs
remain almost identical. No exact generated copies were detected; approximate
memorization is not ruled out.

## Experiment ledger

“Updates” counts generator updates, each accompanied by one discriminator update.
The 10k and 50k checkpoints of a trajectory are not independent trials.

| # | Experiment | Methods / seeds | Budgets | Main observation |
|---|---|---|---|---|
| 1 | Eight-Gaussian ring pilot | Vanilla, paired; seeds 0–4 | 10k | Paired covered all eight modes in every seed; better mean mode balance. |
| 2 | Ring extension | Same methods/seeds | 50k | Paired retained full coverage; vanilla lost modes in several seeds. |
| 3 | Ring relativistic baseline | New RSGAN runs, seeds 0–4; reuse #1–2 | 10k / 50k | Paired had lower mode TV than RSGAN for every seed at both endpoints. |
| 4 | Unconditional CIFAR-10 pilot | Vanilla, RSGAN, paired; seed 0 | 10k | Paired lagged in FID, KID, precision, and recall. |
| 5 | Unconditional CIFAR extension | Resume #4 | 50k | Paired largely caught up, without a clear advantage. |
| 6 | CIFAR, labels in G only | All three; seed 0 | 10k / 50k | G largely ignored labels; same-requested-class sampling did not ensure semantic matching. |
| 7 | CIFAR, labels in G and D | All three; seed 0 | 10k / 50k | More label sensitivity, incomplete adherence; paired behind both baselines at both endpoints. |
| 8 | CIFAR, conditional BatchNorm G + projection D | All three; seed 0 | 10k / 50k | Better FID/KID than #7 for all methods; paired still behind at 50k, with incomplete class adherence. |
| 9 | CIFAR integrity study, same #8 architecture | All three; replay seed 0 | 10k–100k, every 10k | Growing D train/held-out gap; smaller late quality gains; train/held-out FIDs nearly identical; no exact generated copies detected. |

## What the methods mean

- **Vanilla:** unary real/fake discriminator, BCE; G uses non-saturating BCE.
- **RSGAN:** a shared unary critic with logit `C(real) - C(fake)`; D targets 1,
  G targets 0. This is RSGAN, not the batch-average RaSGAN variant.
- **Paired (ours):** unrestricted joint discriminator on concatenated samples in
  randomized A/B slots; BCE predicts whether A is real, and G reverses the label.
  No PairGAN/RelationGAN loss or reconstruction term is used.

Within each comparison, methods receive the same numbers of real and generated
samples. Vanilla makes two unary decisions per pair; paired/RSGAN make one pair
decision. Equal steps do not imply equal FLOPs or equivalent gradient scaling.
A paired discriminator can ignore a slot, but this does not guarantee vanilla's
optimization behavior or performance.

## 1–3. Eight-Gaussian ring

**Question:** Does joint comparison improve mode coverage, and does a separable
relativistic critic provide the same benefit?

The real distribution is an equal mixture of eight isotropic 2-D Gaussians, with
centers equally spaced on a radius-2 circle and per-axis standard deviation 0.1.
G and D are MLPs with two 128-wide hidden layers; latent dimension 16, batch 256,
Adam learning rate 0.0002 and betas (0.5, 0.999), D:G update ratio 1:1. Runs used
CPU with one thread and seeds 0–4. Initial G weights and data/noise streams match
across methods within each seed; vanilla/RSGAN also share initial critic weights.

Evaluation uses 10,000 fixed latent draws every 1,000 updates. A sample is valid
within distance 0.3 of a mode center; a mode is covered when at least 1% of *all*
evaluation samples are valid in that mode.

**Mode TV (total variation distance)** measures how far the accepted mode masses
are from the desired 12.5% per mode, penalizing both uneven allocation and invalid
samples. Let $p_i$ be the fraction of **all** generated samples accepted in mode
$i$, and $p_{\mathrm{invalid}}$ the fraction outside every acceptance region:

$$
\mathrm{ModeTV} = \frac{1}{2}\left(\sum_{i=1}^{8}\left|p_i-\frac18\right| + p_{\mathrm{invalid}}\right).
$$

The fractions are not renormalized after discarding invalid samples. Lower is
better, on a scale from 0 to 1:

| Generated sample allocation | Mode TV |
|---|---:|
| Exactly 12.5% accepted in each mode; none invalid | 0 |
| Equally split across only four modes; none invalid | 0.5 |
| All accepted in one mode | 0.875 |
| All invalid | 1 |

Thus paired's mean 0.069 versus RSGAN's 0.198 at 50k indicates better agreement
with these target masses, including the invalid-sample penalty. It does not
measure how well samples reproduce the Gaussian shape within each mode.
The zero-invalid target is a scoring convention: the real Gaussians themselves
have about 1.1% of their mass beyond the 3-sigma acceptance radius, so even true
distribution samples are not expected to score exactly zero.

Mean ± sample standard deviation across five seeds:

| Updates | Method | Modes covered / 8 ↑ | Valid samples ↑ | Mode TV ↓ |
|---:|---|---:|---:|---:|
| 10k | Vanilla | 7.60 ± 0.89 | 84.9% ± 1.2% | 0.226 ± 0.072 |
| 10k | RSGAN | 7.40 ± 1.34 | 83.6% ± 1.5% | 0.281 ± 0.170 |
| 10k | Paired | 8.00 ± 0.00 | 83.1% ± 0.3% | 0.169 ± 0.003 |
| 50k | Vanilla | 6.80 ± 0.84 | 92.7% ± 1.6% | 0.183 ± 0.074 |
| 50k | RSGAN | 7.40 ± 1.34 | 93.0% ± 1.7% | 0.198 ± 0.220 |
| 50k | Paired | 8.00 ± 0.00 | 93.1% ± 2.0% | 0.069 ± 0.020 |

**Interpretation:** encouraging evidence for coverage and balance in this specific
untuned synthetic setup. At 50k, full coverage occurred in 5/5 paired, 4/5 RSGAN,
and 1/5 vanilla runs. RSGAN's seed 4 retained only five modes, contributing to its
large variability. The result does not identify the mechanism responsible.

**Run lineage:** the vanilla/paired extension replayed training from initialization
and extended it; all 110 original evaluation rows reproduced exactly, excluding
timing. RSGAN trained once to 50k with its 10k endpoint retained. The assembled
three-method comparison reuses those runs, rather than training another suite.

- #1: [Initial report](results/ring8-v1/REPORT.md) · [Config](configs/ring8.json)
- #2: [Extension report and graphics](results/ring8-50k/EXTENSION.md) · [Config](configs/ring8-50k.json)
- #3: [Combined RSGAN comparison](results/ring8-relativistic-comparison/REPORT.md) · [Raw RSGAN report](results/ring8-rsgan-50k/REPORT.md)
- Graphics: [50k samples](results/ring8-relativistic-comparison/50k/final_samples.png) · [Accepted mode masses](results/ring8-50k/mode_mass.png)

## Shared CIFAR-10 protocol

All CIFAR experiments use the 50,000 training images at 32×32 resolution,
normalized to [-1,1], with no augmentation. Training runs on Modal, requesting
one A10 per method, in float32 with TF32 disabled and deterministic algorithms.

The architecture rows below describe #4–7; #8 changes conditioning and the D
output head as described in its section. Training settings remain shared.

| Choice | Setting |
|---|---|
| G | DCGAN-style transposed convolutions; hidden channels 256, 128, 64; ReLU and BatchNorm; tanh RGB output |
| D | Stride-2 convolutions; hidden channels 64, 128, 256; LeakyReLU slope 0.2; scalar logit |
| D normalization | None; no BatchNorm, spectral normalization, or dropout |
| Noise | 128 independent standard-normal coordinates |
| Optimizer | Adam for G and D; both learning rates 0.0002; betas (0.5, 0.999) |
| Batch / update ratio | 128 real and 128 fake per D update; 1 D then 1 G update |
| Sampling | Fresh noise and, where used, fresh real references for G updates |
| Other controls | No LR schedule, weight decay, gradient clipping, gradient penalty, label smoothing, or G weight averaging |
| Initialization | Convolution weights normal with SD 0.02; biases zero |
| G BatchNorm | Running statistics update during both D and G fake-generation passes |
| Replication | Seed 0 only; no method-specific tuning |

**Evaluation:** every reported endpoint uses 10,000 generated images and the same
fixed 10,000-image subset of CIFAR-10's training split. FID and KID use
`torch-fidelity` Inception-2048 features; precision/recall also use these features,
with k=3. KID uses 100 subsets of 1,000 images. Its subset standard deviation is
not uncertainty across training seeds. These are FID-10k evaluations even at
50k *training updates*. Precision/recall are feature-space metrics, not literal
class coverage. Training-set references do not independently test memorization.

Lower FID/KID and higher precision/recall are better. None measures label adherence.
Full implementation details: [base protocol](docs/CIFAR10.md).

## 4–5. Unconditional CIFAR-10

**Question:** Does the toy advantage transfer to unconditional image generation,
and does paired catch up with more training? Neither G nor D receives labels;
real/fake pairs are independently sampled.

| Updates | Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |
|---:|---|---:|---:|---:|---:|
| 10k | Vanilla | 97.14 | 85.91 | 0.661 | 0.042 |
| 10k | RSGAN | 93.83 | 79.62 | 0.618 | 0.030 |
| 10k | Paired | 122.64 | 108.48 | 0.571 | 0.009 |
| 50k | Vanilla | 49.99 | 37.09 | 0.589 | 0.339 |
| 50k | RSGAN | 49.70 | 35.65 | 0.587 | 0.331 |
| 50k | Paired | 53.27 | 35.99 | 0.561 | 0.303 |

**Interpretation:** paired is behind at 10k, then largely catches up by 50k.
At 50k, it remains worse in FID, precision, and recall; its KID is slightly better
than vanilla and close to RSGAN. This is not a consistent paired advantage.

**Run lineage:** the 50k run resumes the original 10k checkpoints. The original
300 logged training rows and 10k generator checkpoints, grids, and metrics were
preserved byte-for-byte. Local directories contain both the original snapshot
and the extension; these are not additional independent runs.

[10k report](results/cifar10-pilot-v1/REPORT.md) · [50k extension](results/cifar10-50k/EXTENSION.md) · [50k samples](results/cifar10-50k/samples_050000.png) · [Config](configs/cifar10.json)

## 6. CIFAR-10: class-matched references, labels in G only

**Question:** Would semantically closer references help paired? Every G receives
a 10-dimensional one-hot label concatenated to its noise. Each batch requests
12 or 13 examples per class; real references are sampled from that requested
class. All three methods use the same conditional G and balanced sampling.
Discriminators still receive images only. Each method trains once to 50k,
retaining the 10k endpoint.

| Updates | Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |
|---:|---|---:|---:|---:|---:|
| 10k | Vanilla | 98.94 | 87.12 | 0.651 | 0.046 |
| 10k | RSGAN | 100.86 | 87.35 | 0.584 | 0.087 |
| 10k | Paired | 119.62 | 107.31 | 0.568 | 0.018 |
| 50k | Vanilla | 51.24 | 37.30 | 0.599 | 0.334 |
| 50k | RSGAN | 48.60 | 32.98 | 0.555 | 0.347 |
| 50k | Paired | 50.59 | 35.20 | 0.567 | 0.303 |

**Interpretation:** the class grids show that generators largely ignore labels.
Consequently, this did not cleanly test semantically close real/fake pairs.
If G ignores the requested label and matches the real marginal distribution,
the image-only paired discriminator cannot penalize that solution. Marginal
scores alone cannot expose this failure. RSGAN has the best 50k FID/KID/recall;
paired has slightly better FID/KID than vanilla but lower precision/recall.

[Report](results/cifar10-classmatched-v1/REPORT.md) · [Class grids](results/cifar10-classmatched-v1/classes_050000.png) · [Protocol](docs/CIFAR10_CLASSMATCHED.md) · [Config](configs/cifar10-classmatched.json)

## 7. CIFAR-10: labels in both G and D

**Question:** Does giving D the requested class close the label-ignoring loophole?
G initialization, class sampling, and optimizer settings match #6. Every D now
also receives ten broadcast one-hot image channels: 13 input channels for unary
critics, 16 for paired. This adds 10,240 weights per D and changes its
initialization. There is no auxiliary classification loss. Each method trains
once to 50k, retaining the 10k endpoint.

| Updates | Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |
|---:|---|---:|---:|---:|---:|
| 10k | Vanilla | 96.54 | 80.25 | 0.657 | 0.062 |
| 10k | RSGAN | 106.52 | 91.28 | 0.570 | 0.021 |
| 10k | Paired | 174.29 | 167.07 | 0.408 | 0.004 |
| 50k | Vanilla | 52.00 | 35.70 | 0.560 | 0.295 |
| 50k | RSGAN | 54.52 | 37.60 | 0.552 | 0.242 |
| 50k | Paired | 61.96 | 45.45 | 0.523 | 0.222 |

**Interpretation:** stronger visible label effects, but still incomplete semantic
class adherence, especially across animal classes. Paired finishes behind both
baselines on all four marginal metrics at both budgets. Its 50k FID worsens from
50.59 in #6 to 61.96, and recall falls from 0.303 to 0.222. Conditioning D does
not yield the hoped-for improvement in this seed; incomplete adherence still
limits the reference-proximity test.

**Hardware note:** Modal supplied an A10G for paired and A10s for vanilla/RSGAN,
despite all jobs requesting A10. Timing is not a controlled same-device comparison.

[Report](results/cifar10-conditional-v1/REPORT.md) · [Class grids](results/cifar10-conditional-v1/classes_050000.png) · [Protocol](docs/CIFAR10_CONDITIONAL.md) · [Config](configs/cifar10-conditional.json)

## 8. CIFAR-10: conditional BatchNorm G and projection D

**Question:** Can stronger architectural access to labels improve adherence and
make same-class references useful? G now uses class-specific BatchNorm scale and
bias in every hidden block, replacing its initial one-hot concatenation. D uses
global sum pooling and scores `linear(h) + embedding(y) · h`, replacing broadcast
label channels and the final spatial convolution. Paired uses joint A/B features
for h. All three methods use these changes consistently, with identical G
initialization within this experiment. BCE, sampling, Adam settings, and evaluation
are unchanged; no auxiliary classifier or spectral normalization is added.

This changes both networks and initialization, not just label strength in an
otherwise identical model. G has 1,191,683 parameters; unary D 661,697 and paired
D 664,769. All three runs used NVIDIA A10. Each ran once to 50k, retaining 10k.

| Updates | Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |
|---:|---|---:|---:|---:|---:|
| 10k | Vanilla | 81.64 | 62.40 | 0.602 | 0.075 |
| 10k | RSGAN | 91.31 | 78.57 | 0.596 | 0.035 |
| 10k | Paired | 121.23 | 106.41 | 0.624 | 0.012 |
| 50k | Vanilla | 41.29 | 27.21 | 0.574 | 0.335 |
| 50k | RSGAN | 45.62 | 28.63 | 0.582 | 0.313 |
| 50k | Paired | 48.33 | 31.36 | 0.570 | 0.287 |

**Interpretation:** all three improve 50k FID/KID relative to #7. Paired's FID
improves from 61.96 to 48.33, but vanilla (41.29) and RSGAN (45.62) remain ahead;
paired also has lower precision/recall than both at 50k. The architectural change
helps marginal quality in this seed but does not show a paired advantage.

Class grids show changes in object identity for some labels, especially vehicles
and horses, but considerable ambiguity across animal classes. Paired's 50k
label-change pixel MAE is 0.1333 versus 0.1311 in #7: its quality improvement does
not establish substantially stronger class control. Pixel sensitivity is not
semantic accuracy; no independent classifier accuracy was measured. The relative
roles of conditional BatchNorm and projection remain unseparated.

[Report](results/cifar10-projection-v1/REPORT.md) · [Class grids](results/cifar10-projection-v1/classes_050000.png) · [Protocol](docs/CIFAR10_PROJECTION.md) · [Config](configs/cifar10-projection.json)

## 9. CIFAR-10 integrity study: held-out data and dense evaluation

**Question:** Are we overfitting, and where do additional training updates stop
paying off? Use the unchanged #8 CBN/projection architecture, losses, optimizer,
and seed 0. Replay from initialization, retain **both G and D every 10k**, and
continue to 100k. Evaluate all ten checkpoints for each method, without early
stopping or checkpoint selection. G weights reproduce #8 exactly at 10k and 50k;
D weights reproduce its 50k models exactly. This extends the same trajectories,
not an independent replication.

**New diagnostics:** generator metrics against the official 10,000-image CIFAR-10
test split, never used in GAN updates, alongside the existing training reference.
Both comparisons use the same 10,000 generated images. D comparisons hold fake
images and labels fixed while swapping class-balanced training versus held-out
real images; paired averages both slot orientations. Nearest-neighbor audits use
pixel and Inception distances, equal 10k candidate pools, plus all 50k training
images for a more complete copying search. Raw neighbor indices and per-example
D diagnostics are retained locally and on Modal.

| Method | Held-out FID 50k → 100k ↓ | Held-out KID ×1000 50k → 100k ↓ | Held-out recall 50k → 100k ↑ | Lowest observed held-out FID |
|---|---:|---:|---:|---|
| vanilla | 41.19 → 38.44 | 27.23 → 23.74 | 0.331 → 0.347 | 38.08 at 80k |
| rsgan | 45.69 → 38.74 | 28.69 → 23.24 | 0.311 → 0.348 | 38.74 at 100k |
| paired | 48.24 → 42.53 | 31.39 → 27.13 | 0.287 → 0.331 | 42.53 at 100k |

**Training duration:** gains are large early, smaller after roughly 50k–70k, and
not monotonic. Vanilla's FID is nearly flat from 70k onward, with its lowest
observed value at 80k. RSGAN and paired still improve to their lowest observed FID
at 100k. This does not establish eventual convergence or a universal stopping
point. Paired remains behind on 100k FID, KID, and recall; precision is slightly
higher than RSGAN's. The observed minima are descriptions, not separately tested
selected checkpoints.

**D overfitting is evident.** At 100k:

| Method | Training real acceptance | Held-out real acceptance | Train-membership AUC from D margin |
|---|---:|---:|---:|
| vanilla | 97.5% | 55.0% | 0.709 |
| rsgan | 96.1% | 77.1% | 0.652 |
| paired | 91.9% | 75.4% | 0.655 |

For vanilla, real-target BCE is 0.099 on training images versus 1.504 on held-out
images. RSGAN compares real/fake margins, and paired compares aligned pair margins;
these acceptance rates are not identical classification tasks across methods.
AUC near 0.5 means no ranking preference; above 0.5 means training examples tend
to receive higher real-acceptance scores. The gap grows with training, while
held-out generator metrics can still improve. This does not prove the D gap
caused a particular generator result.

**Generator memorization is not established.** Train-reference and held-out FID
stay within 0.16 at every checkpoint. No exact generated/training pixel matches
occur at any evaluated checkpoint. At 100k, all methods produce 10,000 distinct
quantized images; equal-pool training-neighbor preference is 50.6–51.7% in
Inception space and 52.8–54.1% in pixel space. Similarity and these modest
preferences do not prove copying; absence of exact copies does not rule out
approximate memorization. Inspect the nearest-neighbor panels as supporting
evidence. Larger training candidate pools naturally yield closer neighbors.

**Calibration:** the real train-10k versus test-10k reference comparison itself
has FID 5.20 and KID approximately zero. No exact pixel duplicates were found
between official train and test images. This does not exclude near-duplicates.
These held-out images are now a diagnostic reference; future tuning based on
them should not be described as evaluation on an untouched final test set.

[Full report](results/cifar10-integrity-v1/REPORT.md) · [Learning curves](results/cifar10-integrity-v1/learning_curves.png) · [D gaps](results/cifar10-integrity-v1/discriminator_gaps.png) · [Nearest neighbors](results/cifar10-integrity-v1/nearest_panel_inception_100k.png) · [Protocol](docs/CIFAR10_INTEGRITY.md)

## Verification and storage

Smoke runs and interrupted-versus-continuous replay checks are engineering
validation, not additional scientific comparisons. The toy extension reproduces
its earlier trajectory. CIFAR implementations have exact CPU/CUDA resume checks
for all methods; the latest implementation passed 53 local tests. Reports check
saved source fingerprints, configuration consistency, finite metrics, and
completion at the requested endpoints.

Reports, plots, small metrics, and source snapshots live under `results/` and
are intended for Git. Large model/optimizer checkpoints are ignored by Git and
retained locally and, for CIFAR, in Modal Volumes. Each new experimental change
gets a separate run directory; old results should remain unchanged.

## Open questions and proposed experiments — not yet run

1. **Further CIFAR adherence experiments:** #8 completed conditional BatchNorm
   and projection together. Possible next tests include an auxiliary class loss
   and separate architectural ablations. Measure semantic accuracy with an
   independently validated classifier before interpreting closer-reference effects.
2. **Toy reference-distance intervention:** keep the target distribution and
   real/fake batches fixed while varying near/random/far associations. Include
   vanilla and RSGAN controls. First check that the pairing rule does not leak
   slot identity when both batches come from the true distribution.
3. **Hyperparameter sensitivity and replication:** G/D learning rates, update
   ratio, capacity, and pair-interaction architecture have not been swept.
   CIFAR conclusions need more seeds before claiming a reliable ranking.

No auxiliary-classification-loss experiment, reference-distance experiment,
dedicated mode-recovery test, or hyperparameter sweep has been completed. Experiment #9 now measures the learning curves every 10k through 100k;
a universal “twice as many steps” convergence penalty is still not established. The mechanism behind the toy advantage remains unresolved.
