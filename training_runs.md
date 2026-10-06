# Training runs

Last updated: 2026-10-05. Running overview of completed research experiments.
Detailed reports, configurations, and raw metrics are linked below. Add future
experiments here, preserving earlier results and distinguishing new trials from
extensions or reused checkpoints.

## Where we stand

**Paired improved mode coverage and balance on the eight-Gaussian toy task across
five seeds. CIFAR-10 fidelity metrics have not shown a consistent paired advantage.** On
CIFAR, paired starts behind and improves with longer training, but has not shown
a consistent FID/KID advantage over vanilla or RSGAN. Retrospective class classification (#12) does find better raw class balance for paired at 50k under two classifiers, with substantial generated-image ambiguity. All CIFAR comparisons have only
one training seed; #10 adds two targeted, untuned D batch/capacity probes.

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

The paired D probes (#10) show a modest gain from doubling only D batch, and a
larger late gain from widening D. At 100k the wider model nearly matches RSGAN
on FID/KID/precision/recall, at greater compute cost and with more D overfitting.

Ordinary MNIST (#11) provides another representation test: paired has lower
digit-frequency TV in all five seeds at 10k and 50k, while vanilla develops
more severe late imbalance. Confidence and sample grids qualify this as a
category-balance result, not an unconditional image-quality win.

The mode-count sweep (#14) qualifies the synthetic results: fixed-radius 16/32-mode
mixtures become easier to cover coarsely as gaps shrink. Fine spatial evaluation
shows that successful mode allocation does not imply correct Gaussian density.

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
| 10 | CIFAR paired D batch/capacity probes | Paired D batch256 and paired D width128; seed 0; reuse #9 baselines | 10k–100k, every 10k | At 100k, held-out FID 41.05 (D batch256), 38.72 (wider D), versus 42.53 paired baseline; G update batch remains 128. |
| 11 | Ordinary MNIST, unconditional | Vanilla and paired; seeds 0–4 | 10k / 50k; evaluation every 1k | Paired lower mode TV in all five seeds at 10k and 50k; 50k full coverage 5/5 paired versus 0/5 vanilla. |
| 12 | Unconditional CIFAR class representation (evaluation only) | Reuse #4–5, all three methods, seed 0; two frozen classifiers | 10k / 50k | Paired has best raw class TV at 50k under both classifiers, worst at 10k; all methods cover ten classes at 50k. Confidence-filtered ranking is mixed. |
| 13 | Conditional CIFAR class representation and adherence (evaluation only) | Reuse #9 original CBN/projection vanilla, RSGAN, paired; seed 0; two classifiers | 10k–100k every 10k | Paired adherence trails both baselines at every checkpoint under both evaluators; no consistent class-TV advantage. |
| 14 | Ring mode-count sweep | 16/32 components, all three methods, seeds 0–4; reuse 8-component runs | 10k / 50k | At 16 modes paired retains a late coarse-TV advantage; at 32 modes coverage saturates for all methods and thin-ring outputs expose the metric’s limitations. |

## What the methods mean

- **Vanilla:** unary real/fake discriminator, BCE; G uses non-saturating BCE.
- **RSGAN:** a shared unary critic with logit `C(real) - C(fake)`; D targets 1,
  G targets 0. This is RSGAN, not the batch-average RaSGAN variant.
- **Paired (ours):** unrestricted joint discriminator on concatenated samples in
  randomized A/B slots; BCE predicts whether A is real, and G reverses the label.
  No PairGAN/RelationGAN loss or reconstruction term is used.

Within comparisons #1–9 and #11, methods receive the same numbers of real and
generated samples per D update. Experiment #10 deliberately doubles D samples in one arm. Vanilla makes two unary decisions per pair; paired/RSGAN make one pair
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

## 10. Paired discriminator batch and capacity probes

Two independent paired arms: D batch256 at width64, and D width128 at batch128.
G stays at width64 and update batch128 with one optimizer step per iteration.
Both use the #9 training/evaluation protocol, running from initialization through
100k with evaluation every 10k. Baselines are reused from #9. See the
[protocol](docs/CIFAR10_CAPACITY.md) for compute accounting and the extra
BatchNorm running-statistic update incurred while producing more D fakes.

Completed 100k updates and all 20 evaluations. [Detailed report](results/cifar10-capacity-v1/REPORT.md).

| Arm | 50k held-out FID | 100k held-out FID | 100k KID ×1000 | 100k precision | 100k recall |
|---|---:|---:|---:|---:|---:|
| vanilla | 41.19 | 38.44 | 23.74 | 0.614 | 0.347 |
| rsgan | 45.69 | 38.74 | 23.24 | 0.580 | 0.348 |
| paired | 48.24 | 42.53 | 27.13 | 0.584 | 0.331 |
| d_batch256 | 49.17 | 41.05 | 25.42 | 0.590 | 0.342 |
| d_width128 | 47.79 | 38.72 | 23.35 | 0.581 | 0.347 |

Larger D batch gives a modest final gain; widening D improves the later trajectory,
nearly matching RSGAN at 100k on FID, KID, precision and recall, with more D
train/held-out separation. These are resource increases, not matched
compute wins. Both preserve G architecture, initial weights, update batch and
optimizer update counts. Extra no-gradient G forwards in the larger-D-batch arm
also update BatchNorm running statistics. One seed, fixed learning rates;
projection scale is not retuned for the wider feature vector.

58 local tests passed; CUDA exact-resume checks passed for both arms and unchanged
settings exactly replayed the previous trainer. All 50 comparison evaluations
use identical references. Training source/configuration, G/D snapshots, loss
logs, neighbor arrays, and sample grids are retained; no automatic commits.

## 11. Ordinary MNIST: vanilla versus paired

Five seeds per method, native 28×28 grayscale images, unconditional training on
all 60,000 training images with their empirical digit proportions. G architecture,
G batch128, D real/fake counts, learning rates and update counts are matched.
Digit labels are used only for evaluation; paired references are independent.

The frozen classifier is selected on a validation subset of training data and
scores 99.14% on official test images. Evaluate 10,000 fixed samples every 1k
updates through 50k, retaining the 10k endpoint as well. Report per-digit mass,
coverage, mode TV, confidence-filtered coverage and randomly selected galleries.
Confidence is an uncalibrated quality proxy, not a validity oracle. See the
[protocol](docs/MNIST.md) and [full report](results/mnist-v1/REPORT.md).

Completed all ten runs and 500 evaluations. Paired has lower frequency TV in all
five matched seeds at both 10k and 50k. At 10k both methods cover all ten digits;
late training produces more severe digit imbalance in vanilla.

| Updates | Method | Covered digits (mean ± SD) | Mode TV (mean ± SD) | Confidence acceptance |
|---|---|---:|---:|---:|
| 10k | vanilla | 10.00 ± 0.00 | 0.140 ± 0.019 | 76.3% |
| 10k | paired | 10.00 ± 0.00 | 0.082 ± 0.011 | 77.3% |
| 50k | vanilla | 8.80 ± 0.45 | 0.341 ± 0.064 | 85.0% |
| 50k | paired | 10.00 ± 0.00 | 0.126 ± 0.008 | 82.5% |

Coverage means at least 1% of all generated examples assigned to a digit; it does
not imply undercovered digits have zero probability. At 50k, all-ten-digit raw
coverage holds in 5/5 paired runs and 0/5 vanilla runs. Confidence-filtered
coverage holds in 5/5 paired and 0/5 vanilla runs. Paired has lower
confidence-augmented TV in all five matched seeds. Overall classifier confidence
is not higher for paired at 50k, so do not equate the balance advantage with an
across-the-board fidelity advantage. Images and per-digit galleries are retained.

63 local tests passed, and CUDA checks confirmed exact checkpoint resume,
identical G initialization, and no change to training caused by evaluation.
Classifier test accuracy is 99.14%; its probabilities are uncalibrated. Results
support a repeatable digit-representation advantage in this configuration, not
a general GAN ranking or a proof of within-digit style diversity.


## 12. Retrospective unconditional CIFAR class representation

No GAN retraining. Re-evaluate #4–5 checkpoints with the original fixed 10,000
noise draws at 10k and 50k. Two frozen CIFAR classifiers (ResNet56 and VGG16-BN)
achieve 94.37% and 94.16% accuracy on official test images. The target class
frequencies are 10% each. Full protocol, threshold sensitivity, sample galleries,
and provenance are in the [report](results/cifar10-class-representation-v1/REPORT.md).

| Endpoint | Classifier | Vanilla class TV | RSGAN class TV | Paired class TV |
|---|---|---:|---:|---:|
| 10k | ResNet56 | 0.4265 | 0.4530 | 0.4874 |
| 10k | VGG16-BN | 0.3523 | 0.3281 | 0.4150 |
| 50k | ResNet56 | 0.2488 | 0.2214 | **0.2038** |
| 50k | VGG16-BN | 0.2163 | 0.1818 | **0.1361** |

Class TV is half the sum of absolute differences between predicted class
frequencies and 10%; lower means more balanced. All methods exceed 1% in all
ten predicted classes at 50k. At 10k, automobile falls below 1% under both
classifiers for all methods. Coverage alone therefore misses the late difference.

The confidence-filtered result is less consistent. At 50k, augmented TV at
confidence ≥0.9 is vanilla/RSGAN/paired = 0.4315/0.4196/0.4296 under ResNet and
0.3546/0.3194/0.2897 under VGG. Paired has lower acceptance under both classifiers.
Accepted mass uses ALL generated images as denominator; rejected mass is included
in augmented TV, exactly as in the MNIST audit.

This supports a late class-balance signal alongside worse paired FID, but only
for one training seed. The evaluators agree on only 55.82–59.24% of generated
images at 50k versus 93.93% of real test images. Galleries contain ambiguous
samples; real-test accuracy and high softmax confidence do not guarantee semantic
accuracy on generated images. No within-class diversity or causal mechanism
claim follows. All 12 records and retained checkpoint/image/weight hashes were
verified; 65 local tests passed. No conditional CIFAR models were re-evaluated.

## 13. Conditional CIFAR class representation and adherence

Re-evaluate the original CBN G / projection D integrity-study checkpoints (#9),
vanilla, RSGAN and paired, seed 0, every 10k through 100k. No GAN training.
D width 64, D batch 128 and G batch 128 throughout; #10 capacity/batch arms are
excluded. The 10,000 generated images at each of 30 checkpoints match the earlier
integrity evaluation byte-for-byte. Requested labels are exactly balanced.
Use the same frozen ResNet56 and VGG16-BN classifiers as #12.

| Updates | Method | Class TV ResNet / VGG ↓ | Requested-class adherence ResNet / VGG ↑ |
|---|---|---|---|
| 50k | Vanilla | 0.1178 / 0.0865 | 64.42% / 66.77% |
| 50k | RSGAN | 0.1268 / 0.1001 | 61.83% / 65.77% |
| 50k | Paired | 0.1492 / 0.0989 | 56.95% / 61.42% |
| 100k | Vanilla | 0.1021 / 0.0747 | 68.89% / 72.07% |
| 100k | RSGAN | 0.0981 / 0.0772 | 68.72% / 71.60% |
| 100k | Paired | 0.1232 / 0.0876 | 64.41% / 67.43% |

Paired has lower requested-class adherence at all ten checkpoints under both
classifiers. Longer training improves all three; paired continues improving
through 100k. Marginal class balance has no consistent paired advantage (there
is a VGG-only paired TV win at 80k). All methods cover ten predicted classes at
50k and 100k. The late unconditional result in #12 does not generalize cleanly
to this conditional setup. Balanced marginal predictions can hide incorrect
conditioning, so requested/predicted confusion matrices are included.

Classifier agreement at 100k is 74.30% vanilla, 74.40% RSGAN and 71.67% paired,
versus 93.93% on real images. Single-seed results and generated-image ambiguity
still limit conclusions. At 100k, the fraction both correct for the request and
confidence ≥0.9 is 59.81% / 66.31% vanilla, 58.93% / 65.74% RSGAN and
54.89% / 61.29% paired (ResNet / VGG); the denominator includes ALL images.

[Full report, curves, confusion matrices and galleries](results/cifar10-conditional-representation-v1/REPORT.md).
All 60 classifier records, checkpoint/image/weight hashes and source snapshot
verified; 67 tests passed. No model or optimizer changes; nothing about how the
discriminator uses its paired input is causally established by this evaluation.

## 14. Ring mode-count sweep: 8 → 16 → 32

30 new unconditional runs: vanilla, RSGAN and paired × five seeds × 16/32
components. Reuse the existing 8-component runs. Keep radius 2, sigma 0.1,
networks, batch 256, Adam settings and 50k budget unchanged. Retain the 10k
endpoint and evaluate mode metrics every 1k. Changing component count also
changes separation and expected per-component samples per batch; this is a
fixed-geometry stress test, not an isolated count intervention.

Mean ± sample SD, five seeds, at 50k:

| Components | Method | Original mode TV ↓ | Fine-grid distribution TV ↓ | Original 1% coverage |
|---|---|---|---|---|
| 8 | Vanilla | 0.183 ± 0.074 | 0.568 ± 0.021 | 6.8/8 |
| 8 | RSGAN | 0.198 ± 0.220 | 0.540 ± 0.150 | 7.4/8 |
| 8 | Paired | 0.069 ± 0.020 | 0.555 ± 0.043 | 8/8 |
| 16 | Vanilla | 0.171 ± 0.089 | 0.539 ± 0.042 | 16/16 |
| 16 | RSGAN | 0.165 ± 0.102 | 0.534 ± 0.049 | 16/16 |
| 16 | Paired | 0.088 ± 0.010 | 0.568 ± 0.033 | 16/16 |
| 32 | Vanilla | 0.038 ± 0.011 | 0.581 ± 0.110 | 32/32 |
| 32 | RSGAN | 0.035 ± 0.012 | 0.554 ± 0.070 | 32/32 |
| 32 | Paired | 0.053 ± 0.019 | 0.526 ± 0.055 | 32/32 |

**Coverage is misleading at high density.** Adjacent means at 32 components
are 0.392 apart, while 3-sigma acceptance disks have radius 0.3 and overlap.
A continuous noisy ring, without the specified discrete Gaussian structure,
scores mode TV 0.0257. All methods cover all 16/32 modes at both endpoints,
even under the original 1% cutoff. A target-relative threshold (8% of each
component’s ideal mass) is additionally retained to avoid making the cutoff
proportionally stricter with increasing count; it does not change this conclusion.

Fine-grid TV compares empirical mass in fixed 0.05×0.05 spatial cells against
exact Gaussian-mixture cell probabilities, including an outside category.
It tests placement and spread within/between modes as well as coarse mass.
Its finite-sample real-distribution reference is 0.111 / 0.155 / 0.192 for
8 / 16 / 32 components (200 independent 10,000-point draws). Scores across
component counts need this reference; it is not a training confidence interval.
Sensitivity to cell widths 0.025 and 0.1 is also saved.

Paired keeps a 16-mode late coarse-TV advantage, but has worse mean fine-grid TV.
At 32 modes it has better mean fine-grid TV but worse coarse TV, and all methods
are far from the real-distribution reference. Plots show thin-ring behavior and
imperfect Gaussian spread. Mean fine-grid TV worsens from 10k to 50k for every
method/count combination at the primary resolution. Retrospectively, the original
8-mode success is a mode-allocation result, not a demonstrated full-density fit.

[Full report, all-seed samples, controls and plots](results/ring-mode-count-v1/REPORT.md).
Configs: [16 modes](configs/ring16-50k.json), [32 modes](configs/ring32-50k.json).
All 90 endpoints recomputed from saved samples; source snapshots and sample
hashes verified. 69 tests passed. No architecture/objective changes or additional
separation-preserving runs.

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
