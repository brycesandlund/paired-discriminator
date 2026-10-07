# Training runs

Last updated: 2026-10-06. Running overview of completed research experiments.
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

The fixed-spacing grid sweep (#15) removes the shrinking-gap issue: paired
strongly improves 9/25-mode representation at 50k across all seeds, but its
49-mode outputs remain too diffuse to beat the baselines on mean TV at 50k.
The exact continuation (#16) reverses that ranking: paired is ahead in all
five seeds on mode TV by 100k and on both TV measures at 150k.
PacGAN2 (#17) closely matches paired’s learning curves and final mode balance.
The gains on these grids therefore do not require a real–fake reference pair;
same-source packing achieves similar improvements. Neither is consistently better
across metrics, and both retain substantial within-mode density errors.
The two-output discriminator (#18) retains those mode-TV gains and improves
late 7×7 fine-grid density fit, but has worse 5×5 density fit than paired/PacGAN2.
The gains therefore also survive separate slot-classification objectives; the
mechanism and a universally best joint-input objective remain unresolved.
Doubling only D’s batch (#19) yields modest early gains but no general speedup;
all three joint methods have worse final mean mode TV on 5×5/7×7. More D
examples per update do not substitute for more updates, and the experiment
does not establish the mechanism or convergence plateaus.
D-only reference interventions (#20) find that adaptive missing-mode sampling
improves both vanilla and paired on the larger grids. Paired reaches all 49
modes at the original threshold in all five seeds, but one adaptive 3×3 seed
fails. D-only nearby matching performs poorly; G still sees random references,
so this does not test matching consistently in both phases.

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
| 15 | Fixed-spacing Gaussian grids | 3×3 / 5×5 / 7×7; vanilla, RSGAN, paired; seeds 0–4 | 10k / 50k | Paired wins mode and fine-grid TV in every seed at 50k on 9/25 modes; at 49 modes its broad coverage has low valid mass and worse mean TV. |
| 16 | 7×7 exact continuation | Continue #15 all three methods, seeds 0–4; restore model/Adam/RNG state | 50k → 100k → 150k | Paired overtakes both baselines: lower mode TV in all seeds at 100k/150k, lower fine-grid TV in all seeds at 150k. |
| 17 | PacGAN2 on fixed-spacing grids | New PacGAN2, seeds 0–4; reuse #15–16 baselines | 10k / 50k; 7×7 also 100k / 150k | Closely tracks paired; final 7×7 mode TV 0.229 vs 0.231 paired. No consistent winner across coarse and fine density metrics. |
| 18 | Two-output joint discriminator | New dual_slot, seeds 0–4; reuse #15–17 | 10k / 50k; 7×7 also 100k / 150k | Retains joint-input mode-TV gains; at 7×7/150k fine-grid TV 0.595 vs 0.647 paired and 0.655 PacGAN2, but 5×5 density fit is worse than both. |
| 19 | D-only batch doubling on grids | Paired, PacGAN2, two-output D512/G256; seeds 0–4; reuse #15–18 | 3×3/5×5 to 50k; 7×7 to 150k; equal-G and equal-D-sample comparisons | Modest early gains do not persist; all three have worse final mean mode TV on 5×5/7×7. Equal-D-sample mode TV worsens versus D256 in all 24 comparisons. |
| 20 | D-only grid reference selection | Paired near, paired deficit, vanilla deficit; seeds 0–4; reuse baselines | 3×3/5×5 to 50k; 7×7 to 150k | D-only near matching fails; deficit sampling helps vanilla and paired on larger grids. 7×7 paired mode TV 0.231→0.183; one paired 3×3 seed fails. |
| 21 | Unconditional CIFAR class deficit sampling | Vanilla and paired, uniform vs adaptive real D sampling; seed 0 | 100k, held-out evaluation every 10k | Completed: deficit improves final class TV for both; paired deficit FID 50.85 vs 48.79 original. Vanilla arms deteriorate after 60k; seed 0 only. |
| 22 | G-only grid deficit references | Paired; seeds 0–4; reuse earlier grid baselines | 3×3/5×5 to 50k; 7×7 to 150k | G-only does not reproduce D-only gains: 7×7 mode TV 0.235 vs 0.231 uniform and 0.183 D-only; full coverage 1/5 vs 1/5 and 5/5. |

## What the methods mean

- **Vanilla:** unary real/fake discriminator, BCE; G uses non-saturating BCE.
- **RSGAN:** a shared unary critic with logit `C(real) - C(fake)`; D targets 1,
  G targets 0. This is RSGAN, not the batch-average RaSGAN variant.
- **Paired (ours):** unrestricted joint discriminator on concatenated samples in
  randomized A/B slots; BCE predicts whether A is real, and G reverses the label.
  No PairGAN/RelationGAN loss or reconstruction term is used.
- **PacGAN2 (#17):** the same joint D architecture as paired, but each input
  contains two independent reals or two independent fakes. D predicts real pack
  versus fake pack; G targets real and receives gradients through both members.

Within comparisons #1–9 and #11, methods receive the same numbers of real and
generated samples per D update. Experiment #10 deliberately doubles D samples in one arm. Vanilla makes two unary decisions per pair; paired/RSGAN make one pair
decision. Equal steps do not imply equal FLOPs or equivalent gradient scaling.
A paired discriminator can ignore a slot, but this does not guarantee vanilla's
optimization behavior or performance.

The two-output arm (#18) jointly processes two points but predicts each slot’s
real/fake identity with its own logit. D uses equal RR/RF/FR/FF counts; G uses
FF/FR/RF and applies BCE only to generated-slot outputs. Every generated point
is used once per phase, including both FF members. This changes the task while
retaining the joint hidden architecture and sample budgets (129 extra D parameters).

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

## 15. Fixed-spacing Gaussian grids: 9, 25 and 49 modes

45 new unconditional runs, all three methods and five seeds per grid size.
Gaussian centers form centered 3×3, 5×5 and 7×7 grids, fixed spacing **1.5**,
sigma **0.1**. Coordinate extents are ±1.5, ±3.0 and ±4.5; the grids expand
without normalization. Adjacent centers remain 15 sigma apart and 3-sigma
acceptance disks never overlap. Larger grids contain all smaller-grid centers.

The original training loop, initialization, models and BCE objectives are
unchanged; automated source comparisons verify this. Latent dimension 16,
G/D width 128, batch 256, Adam 0.0002 (0.5,0.999), one D and one G update per
step, 50k updates, fixed 10,000-sample evaluation. Grid size is the only changed
training configuration between these runs. There is no method-specific tuning.

50k, mean ± sample SD across five seeds:

| Modes | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid mass | Coverage ≥1% of all samples |
|---|---|---|---|---|---|
| 9 | Vanilla | 0.542 ± 0.001 | 0.679 ± 0.031 | 94.9% | 4.0/9 |
| 9 | RSGAN | 0.540 ± 0.002 | 0.703 ± 0.022 | 94.3% | 4.0/9 |
| 9 | Paired | 0.058 ± 0.006 | 0.621 ± 0.010 | 94.3% | 9.0/9 |
| 25 | Vanilla | 0.718 ± 0.024 | 0.794 ± 0.015 | 73.7% | 7.2/25 |
| 25 | RSGAN | 0.686 ± 0.039 | 0.787 ± 0.020 | 73.5% | 7.8/25 |
| 25 | Paired | 0.242 ± 0.050 | 0.597 ± 0.034 | 75.8% | 24.4/25 |
| 49 | Vanilla | 0.667 ± 0.017 | 0.813 ± 0.035 | 37.2% | 11.8/49 |
| 49 | RSGAN | 0.668 ± 0.024 | 0.808 ± 0.007 | 38.8% | 12.8/49 |
| 49 | Paired | 0.710 ± 0.028 | 0.829 ± 0.019 | 29.1% | 6.8/49 |

At 50k, paired beats both baselines on mode TV AND fine-grid TV in every matched
seed on the 9- and 25-mode grids. On 9 modes, the baselines concentrate on four
corners in all seeds while paired covers all nine. On 25 modes, paired reaches
all 25 above the original 1% cutoff in four seeds; seed 4 reaches 22. At the
fixed target-relative threshold (8% of each mode's ideal mass), paired covers
all 25 in every seed. Its 25-mode advantage is late: at 10k it trails the
baselines on mean mode TV and fine-grid TV.

At 49 modes, paired reaches almost all neighborhoods at the looser relative
threshold (48.4/49 on average), but only 29.1% of samples are valid. Broad
spatial support is not equivalent to concentrating the correct mass at Gaussian
centers. Original-threshold coverage averages only 6.8/49, and both TV means
are worse than the baselines. Its mode TV is still improving at every 10k
endpoint: 0.856 → 0.802 → 0.780 → 0.756 → 0.710. The agreed 50k budget does
not establish a plateau or eventual ranking; no further training was run.

Fine-grid TV uses fixed 0.05 cells over [-5.5,5.5]² plus an outside category,
compared with exact Gaussian-mixture cell probabilities. Real-sample references
are 0.117 / 0.191 / 0.264 for 9 / 25 / 49 modes, from 200 draws of 10,000
points. All methods have substantial remaining density mismatch. The mean
50k method rankings are stable at cell widths 0.025, 0.05 and 0.1.
The original 1% coverage threshold becomes stricter relative to ideal mass as
K grows; both it and the constant 8%-of-target threshold are reported.

[Full report, curves, all-seed samples and spatial mass maps](results/grid-mode-count-v1/REPORT.md).
Configs: [3×3](configs/grid3-50k.json), [5×5](configs/grid5-50k.json),
[7×7](configs/grid7-50k.json). All 90 endpoints recomputed and checked against
saved logs; source snapshots and sample hashes verified. 73 tests passed.
Fixed separation does not fix global coordinate range, modes per batch or
capacity per mode; these results apply to the explicitly unscaled expansion.

## 16. 7×7 continuation: another 100k updates

Continue all fifteen #15 7×7 runs from 50k to total 150k, evaluating 100k and
150k. Restore G, D, both Adam states and all explicit RNG streams; preserve the
same fixed 10,000-point evaluation noise. No changes to geometry, networks,
learning rates, batches or losses. Original 50k files remain unchanged.

Mean ± sample SD over five seeds:

| Total updates | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid mass | Coverage ≥1% |
|---|---|---|---|---|---|
| 50k | Vanilla | 0.667 ± 0.017 | 0.813 ± 0.035 | 37.2% | 11.8/49 |
| 50k | RSGAN | 0.668 ± 0.024 | 0.808 ± 0.007 | 38.8% | 12.8/49 |
| 50k | Paired | 0.710 ± 0.028 | 0.829 ± 0.019 | 29.1% | 6.8/49 |
| 100k | Vanilla | 0.564 ± 0.035 | 0.739 ± 0.025 | 66.8% | 18.2/49 |
| 100k | RSGAN | 0.578 ± 0.022 | 0.780 ± 0.013 | 66.1% | 16.8/49 |
| 100k | Paired | 0.352 ± 0.062 | 0.658 ± 0.070 | 65.6% | 33.6/49 |
| 150k | Vanilla | 0.520 ± 0.042 | 0.723 ± 0.026 | 76.4% | 21.0/49 |
| 150k | RSGAN | 0.519 ± 0.056 | 0.726 ± 0.018 | 74.8% | 21.0/49 |
| 150k | Paired | 0.231 ± 0.045 | 0.647 ± 0.058 | 78.2% | 41.4/49 |

The ranking reverses. Paired wins mode TV against both baselines in all five
matched seeds at 100k and 150k. At 150k it also wins fine-grid TV against both
in every seed (at 100k, 4/5 versus vanilla and 5/5 versus RSGAN). Similar valid
fractions at 150k accompany markedly better mode allocation for paired.
At the relative threshold of 8% of target mode mass, coverage is 47.8/49 paired,
34.4/49 vanilla and 35.0/49 RSGAN. Original 1% coverage reaches all 49 modes in
only one paired seed; recovery is not complete. Fine-grid TV also remains far
above the real-draw reference of 0.264, with imperfect spread and bridges in
sample plots. Paired's fine-grid metric improves only slightly after 100k even
as its coarse mode TV improves further.

[Full report, curves, mass maps and all-seed samples](results/grid7-extension-comparison-v1/REPORT.md).
Config: [grid7-150k.json](configs/grid7-150k.json). Extension-only timing is stored
in new logs. Full checkpoints at 100k and 150k are retained locally; source,
plots, metrics and small sample arrays are kept for Git. Continuous-versus-split
training matches bit-for-bit for all three methods, including optimizer and
RNG state. All 15 real resume boundaries verified; all 45 reported endpoints
(including reused 50k) regenerate samples exactly from checkpoints and match
recomputed metrics. 74 local tests passed. No training beyond 150k was run.

## 17. PacGAN2 grid comparison

**Question:** Does same-source packing recover the gains of our mixed real–fake
pairs under the agreed sample and architecture controls?

Fifteen new runs, seeds 0–4: 3×3/5×5 to 50k, 7×7 to 150k. Reuse #15–16
baselines; retain 10k/50k and additionally 100k/150k for 7×7. Same grid spacing,
sigma, G, optimizers, sampling streams and evaluation protocol. PacGAN2 shares
paired’s exact 4→128→128→1 D architecture and initialization. Each D update sees
128 independent real-real packs and 128 fake-fake packs: 256 real points, 256 fake
points and 256 BCE decisions. G uses 256 fresh generated points in 128 packs,
with non-saturating BCE and gradients through both members. Every step retains
one D and one G update. PacGAN2 uses fewer D input rows during G training than
paired, so total compute is not exactly matched. No Gaussian-mode matching.

Final results, mean ± sample SD across five seeds; lower TV is better:

| Grid / budget | Method | Mode TV | Fine-grid TV | Valid | Coverage ≥1% |
|---|---|---:|---:|---:|---:|
| 3×3 / 50k | vanilla | 0.542 ± 0.001 | 0.679 ± 0.031 | 94.9% | 4.0/9 |
| 3×3 / 50k | rsgan | 0.540 ± 0.002 | 0.703 ± 0.022 | 94.3% | 4.0/9 |
| 3×3 / 50k | paired | 0.058 ± 0.006 | 0.621 ± 0.010 | 94.3% | 9.0/9 |
| 3×3 / 50k | pacgan2 | 0.064 ± 0.010 | 0.599 ± 0.050 | 93.7% | 9.0/9 |
| 5×5 / 50k | vanilla | 0.718 ± 0.023 | 0.794 ± 0.015 | 73.7% | 7.2/25 |
| 5×5 / 50k | rsgan | 0.686 ± 0.039 | 0.787 ± 0.020 | 73.5% | 7.8/25 |
| 5×5 / 50k | paired | 0.242 ± 0.050 | 0.597 ± 0.034 | 75.8% | 24.4/25 |
| 5×5 / 50k | pacgan2 | 0.216 ± 0.034 | 0.622 ± 0.028 | 78.4% | 24.6/25 |
| 7×7 / 150k | vanilla | 0.520 ± 0.042 | 0.723 ± 0.026 | 76.3% | 21.0/49 |
| 7×7 / 150k | rsgan | 0.519 ± 0.056 | 0.726 ± 0.018 | 74.8% | 21.0/49 |
| 7×7 / 150k | paired | 0.231 ± 0.045 | 0.647 ± 0.058 | 78.2% | 41.4/49 |
| 7×7 / 150k | pacgan2 | 0.229 ± 0.042 | 0.655 ± 0.048 | 77.7% | 43.2/49 |

PacGAN2 closely tracks paired’s learning curves, including the slower early progress on larger grids. Both strongly improve final mode allocation over vanilla and RSGAN. There is no consistent overall winner between PacGAN2 and paired across these metrics and budgets.

At 50k on 3×3, mode TV is 0.064 ± 0.010 PacGAN2 versus 0.058 ± 0.006 paired; both cover all nine modes under both thresholds in every seed. Fine-grid TV slightly favors PacGAN2 on average (0.599 versus 0.621), with mixed seed-level outcomes. On 5×5, PacGAN2 improves mode TV (0.216 versus 0.242, winning 4/5 matched seeds), while paired improves fine-grid TV (0.597 versus 0.622, winning 4/5). Both recover all 25 modes at the relative threshold in every seed, and at the original 1% threshold in four of five seeds.

On 7×7, PacGAN2 also starts behind the unary baselines, then overtakes them with more training. At 150k its mode TV is 0.229 ± 0.042 versus paired’s 0.231 ± 0.045, vanilla’s 0.520 and RSGAN’s 0.519. The near-equal mean hides seed variation: paired has lower mode TV in 4/5 matched seeds, while PacGAN2’s larger win on seed 0 offsets those differences. Fine-grid TV is 0.655 PacGAN2 versus 0.647 paired, both far above the true-mixture sampling reference of 0.264. Original-threshold coverage averages 43.2/49 versus 41.4/49; each reaches all 49 in only one seed. Samples still show narrow clusters and bridges.

Interpretation: the gains observed here do not require our real–fake reference arrangement; same-source packing produces very similar improvements with the same joint D architecture. This does not establish an identical mechanism or rule out benefits on other tasks. It makes PacGAN2 a necessary baseline for subsequent claims. No method-specific tuning was performed, and the G-step discriminator workload is smaller for PacGAN2.

[Full report, learning curves, per-mode mass and all-seed sample panels](results/grid-pacgan2-comparison-v1/REPORT.md).
All 160 endpoints recomputed and checked against logs; 115 checkpoint replays
match exactly, including all 40 PacGAN2 endpoints. The 45 original baseline
10k arrays have no retained matching checkpoints. Model optimizer step counters
match budgets, and all six RNG states match paired at all 25 mutually retained
checkpoints. All 78 tests passed. Small arrays, plots, logs in CSV and source
snapshots are retained for Git; full checkpoints remain locally ignored.

## 18. Two-output discriminator on fixed-spacing grids

**Question:** Does jointly classifying both inputs retain the mode-coverage gains
when the objective does not require a relative or same-source-pack decision?

Fifteen new trajectories: seeds 0–4 on 3×3/5×5 to 50k, and 7×7 to 150k.
Reuse all four baseline methods from #15–17. Keep G, hidden D layers, Adam,
learning rates, sample budgets, grid geometry and evaluation noise unchanged.
D now has two outputs, one per slot: 4→128→128→2 (17,410 parameters, 129 more
than paired/PacGAN2). Hidden D initialization and all G initialization match
those joint methods for each seed.

D receives 64 RR, 64 RF, 64 FR and 64 FF pairs: 256 real and 256 fake samples,
512 BCE slot decisions. G receives 64 FF, 64 FR and 64 RF pairs: 256 generated
samples with gradients, 128 real references. G averages losses only over its
256 generated-slot predictions; both FF outputs contribute. Every generated
sample is used exactly once per phase. Draw and discard the other 128 real-G
samples and retain unused slot draws to keep RNG consumption identical. One
D and one G update per step. D processes 192 pair rows during G training,
versus 256 paired and 128 PacGAN2; total compute is not exactly matched.

Final results, mean ± sample SD over five seeds:

| Grid / budget | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid | Coverage ≥1% |
|---|---|---:|---:|---:|---:|
| 3×3 / 50k | vanilla | 0.542 ± 0.001 | 0.679 ± 0.031 | 94.9% | 4.0/9 |
| 3×3 / 50k | rsgan | 0.540 ± 0.002 | 0.703 ± 0.022 | 94.3% | 4.0/9 |
| 3×3 / 50k | paired | 0.058 ± 0.006 | 0.621 ± 0.010 | 94.3% | 9.0/9 |
| 3×3 / 50k | pacgan2 | 0.064 ± 0.010 | 0.599 ± 0.050 | 93.7% | 9.0/9 |
| 3×3 / 50k | dual_slot | 0.073 ± 0.032 | 0.602 ± 0.052 | 92.9% | 9.0/9 |
| 5×5 / 50k | vanilla | 0.718 ± 0.023 | 0.794 ± 0.015 | 73.7% | 7.2/25 |
| 5×5 / 50k | rsgan | 0.686 ± 0.039 | 0.787 ± 0.020 | 73.5% | 7.8/25 |
| 5×5 / 50k | paired | 0.242 ± 0.050 | 0.597 ± 0.034 | 75.8% | 24.4/25 |
| 5×5 / 50k | pacgan2 | 0.216 ± 0.034 | 0.622 ± 0.028 | 78.4% | 24.6/25 |
| 5×5 / 50k | dual_slot | 0.211 ± 0.026 | 0.647 ± 0.035 | 79.0% | 24.0/25 |
| 7×7 / 150k | vanilla | 0.520 ± 0.042 | 0.723 ± 0.026 | 76.3% | 21.0/49 |
| 7×7 / 150k | rsgan | 0.519 ± 0.056 | 0.726 ± 0.018 | 74.8% | 21.0/49 |
| 7×7 / 150k | paired | 0.231 ± 0.045 | 0.647 ± 0.058 | 78.2% | 41.4/49 |
| 7×7 / 150k | pacgan2 | 0.229 ± 0.042 | 0.655 ± 0.048 | 77.7% | 43.2/49 |
| 7×7 / 150k | dual_slot | 0.209 ± 0.036 | 0.595 ± 0.059 | 80.1% | 44.8/49 |

The two-output model retains the large final mode-TV advantage over vanilla and RSGAN in every seed on all three grids. Its learning curves broadly track paired and PacGAN2, including slow early progress on the larger grids. It does not establish a consistent winner among the three joint-input methods across grids, budgets and metrics.

At 50k on 3×3, mode TV is 0.073 ± 0.032, compared with 0.058 paired and 0.064 PacGAN2. All five seeds cover all nine modes; one seed has notably worse validity, widening the variation. On 5×5, mean mode TV is 0.211 versus 0.242 paired and 0.216 PacGAN2, but fine-grid TV is worse: 0.647 versus 0.597 and 0.622. Paired wins fine-grid TV in all five matched seeds there. Two-output covers all 25 modes under the relative threshold in every seed, and under the original 1% threshold in four seeds (20/25 in seed 2).

The strongest new result is 7×7 at 150k. Mode TV is 0.209 ± 0.036, versus 0.231 paired and 0.229 PacGAN2. Fine-grid TV is 0.595 ± 0.059, versus 0.647 ± 0.058 paired and 0.655 ± 0.049 PacGAN2: two-output wins this metric in 4/5 matched seeds against paired and 5/5 against PacGAN2. It wins mode TV in 4/5 and 3/5, respectively. Validity is 80.1%, and original-threshold coverage averages 44.8/49 (all 49 in two seeds), versus 41.4 paired and 43.2 PacGAN2. However, lenient relative coverage averages only 46.6/49 versus 47.8 and 48.4: more substantial mass in represented modes does not mean fewer nearly absent modes. Full density recovery remains incomplete; fine-grid TV is still well above the real-sample reference of 0.264, and sample panels show narrow clusters and bridges.

Interpretation: the benefits in this suite survive replacing the relative/pack-level decision with separate per-slot classification. This supports investigating joint-input training more broadly, rather than attributing all gains to a particular comparison objective. It does not isolate the mechanism: shared features, gradient interactions and different compute remain coupled, and no hyperparameters were tuned. The late 7×7 density improvement warrants replication rather than a universal superiority claim.


[Full report, curves, mass maps and all-seed samples](results/grid-dual-slot-comparison-v1/REPORT.md).
All 200 endpoint metrics verified against saved arrays; 155 exact checkpoint
replays, including all 40 new two-output endpoints. Optimizer step counts match
budgets and all six RNG states match paired at all 25 mutually retained
checkpoints. Source hashes and configurations verified. 84 tests passed,
including exact sample accounting, target labels, fake detachment and G gradient
masking. Small arrays, metrics, figures and source snapshots are retained for Git;
full model/optimizer checkpoints remain locally ignored.

## 19. Double only D’s batch on the grids

**Question:** Does more D training data per update remove the joint models’
slower early learning, while leaving G’s workload unchanged?

45 new trajectories: paired, PacGAN2 and two-output, seeds 0–4, on 3×3/5×5
through 50k and 7×7 through 150k. D now receives 512 real and 512 generated
points per update (512 mixed pairs for paired, 256 RR + 256 FF for PacGAN2,
and 128 of each source combination for two-output). G still receives 256
fresh generated samples, with the original method-specific losses and reference
pairing. Same architectures, initialization, Adam learning rate and betas;
one D and one G optimizer update per step. Mean loss reduction is unchanged.
G’s no-gradient forward pass for D produces 512 points instead of 256; its
optimization batch stays fixed.
Eight CPU workers, one torch thread each; existing baselines are reused.

G’s real/noise/slot/evaluation random streams are preserved exactly. D’s larger
batch consumes more samples; a separate D-slot RNG avoids changing G’s slot
assignments. This changes D exposure, gradient noise and compute together.
It does not isolate network capacity or the complexity of the learned function.

Both accounting schemes are retained. At equal G updates, D512 sees twice the
D samples. At equal D samples, D512 runs half as many D/G optimizer updates
and trains G on half as many examples. Neither matches total FLOPs or wall time.
Extra 5k/25k/75k checkpoints allow exact saved-output comparisons at those budgets.

The table below shows mean mode TV (five seeds); lower is better. Reference
budget is 50k for 3×3/5×5 and 150k for 7×7. SDs and all other metrics are in the
full report.

| Grid / reference budget | Method | D256 | D512, equal G updates | D512, equal D samples |
|---|---|---:|---:|---:|
| 3×3 / 50k | paired | 0.058 | 0.060 | 0.089 |
| 3×3 / 50k | pacgan2 | 0.064 | 0.059 | 0.095 |
| 3×3 / 50k | dual_slot | 0.073 | 0.065 | 0.091 |
| 5×5 / 50k | paired | 0.242 | 0.258 | 0.601 |
| 5×5 / 50k | pacgan2 | 0.216 | 0.271 | 0.578 |
| 5×5 / 50k | dual_slot | 0.211 | 0.243 | 0.580 |
| 7×7 / 150k | paired | 0.231 | 0.286 | 0.507 |
| 7×7 / 150k | pacgan2 | 0.229 | 0.248 | 0.512 |
| 7×7 / 150k | dual_slot | 0.209 | 0.281 | 0.529 |

Doubling D’s batch does not remove the joint models’ characteristic early disadvantage or produce a reliable speedup. There are modest early improvements on the larger grids, but no consistent benefit at the final budgets. The hypothesis that simply giving D more examples per update would fix the lag is not supported by this intervention.

At equal G updates on 5×5 at 10k, D512 improves mean mode TV from 0.728→0.691 (paired), 0.741→0.696 (PacGAN2), and 0.743→0.703 (two-output), but all remain behind vanilla/RSGAN at 0.657/0.659. Original-threshold coverage is 13.6/14.2/13.6 modes, versus 15.2/14.6 for unary methods. On 7×7 at 50k there is a qualification: paired D512 reaches 0.659 mode TV, slightly better than vanilla/RSGAN at 0.667/0.668, while PacGAN2/two-output remain behind at 0.692/0.704. Paired’s fine-grid TV and original-threshold coverage still trail the unary baselines there. Thus the early gap narrows in some settings rather than remaining completely unchanged.

At the final budgets, all three D512 arms have worse mean mode TV than their own D256 baselines on both 5×5 and 7×7. At 5×5/50k the changes are 0.242→0.258 paired, 0.216→0.271 PacGAN2, and 0.211→0.243 two-output; each loses in 4/5 matched seeds. At 7×7/150k they are 0.231→0.286, 0.229→0.248, and 0.209→0.281. Fine-grid TV at that endpoint changes from 0.647→0.696, 0.655→0.656, and 0.595→0.665; paired and two-output lose on fine-grid TV in all five matched seeds. Original-threshold coverage falls from 41.4→35.2, 43.2→38.6, and 44.8→34.6 of 49 modes. All three still beat vanilla/RSGAN on mean final mode TV. On 3×3 the result is mixed and all runs cover all nine modes; PacGAN2 improves mean fine-grid TV while two-output worsens it.

At equal cumulative D samples, mean mode TV is worse with D512 in all 24 reported method/grid/reference-budget combinations versus the same D256 method. This comparison gives D512 half as many G/D optimizer updates, so it tests sample efficiency rather than equal optimization effort. For the 7×7 budget corresponding to D256 at 150k, D512 runs 75k and scores 0.507/0.512/0.529, versus 0.231/0.229/0.209 for its D256 counterparts. More D examples in fewer updates do not substitute for the original training trajectory.

Interpretation: the slower start and stronger late mode allocation of joint models largely survive the intervention; extra D batch data does not explain them away. A more difficult learned function remains a plausible hypothesis, not an established mechanism. Batch size changes gradient noise and optimization dynamics while leaving representational capacity fixed; this result does not rule out capacity limitations. The same learning rate was intentionally retained, so these findings concern this controlled change rather than the best achievable larger-batch configuration. These are finite-budget endpoints, not demonstrated convergence plateaus, and none establishes full density recovery.


[Full report, both budget comparisons, samples and mass maps](results/grid-d512-comparison-v1/REPORT.md).
All 425 retained endpoints verified, including 225 new sample arrays that
replay exactly from checkpoints and 200 reused baseline endpoints whose hashes
match the previous report. G RNG states match at all 75 mutually retained
checkpoints. Source hashes, configurations and optimizer step counts verified.
90 tests passed, including exact old-trainer replay when the D batch is unchanged
and checks that only D’s workload doubles. Models and optimizer checkpoints
remain locally ignored; small sample arrays, metrics, figures and source are
retained for Git.

## 20. D-only nearby and underrepresented-mode references

**Question:** Does pairing G outputs with nearby real samples or emphasizing
real samples from missing modes improve grid learning?

45 new runs: paired-near, paired-deficit, vanilla-deficit × seeds 0–4 ×
3×3/5×5/7×7. Original D256/G256, models, optimizer settings and budgets:
50k on smaller grids, 150k on 7×7, with 10k/50k and 100k/150k endpoints.
Only D's real sampling/pairing changes; G still uses independent uniform random
references. Reuse previous random paired/vanilla and other baseline results.

Nearby: exact minimum-total-squared-distance one-to-one assignment between
sampled real and fake batches. Preserve both marginals and all 256 points from
each source. Deficit: EMA of accepted mode mass (decay .99, initialized at 1/K),
updated from the existing D fake batch. Sample real modes with half uniform
probability plus half normalized positive deficit from 1/K, using the previous
EMA. Every mode retains at least .5/K probability. Both paired and vanilla use
the same adaptive algorithm, driven by their own generators; their actual weight
trajectories differ. Evaluate every arm against the original uniform mixture.
[Full prespecified protocol](docs/GRID_REFERENCE_PROTOCOL.md).

Nearby changes association while deficit changes the real training distribution.
These are distinct interventions, not a pure distance-only comparison. D-near
versus G-random contexts also differ, so a negative near result cannot reject
nearby pairing in both phases. A structural source-swap test and learned real-real
audit check accidental source-role information: held-out accuracy is 49.0%,
50.0%, 49.7% across three seeds, consistent with chance at the measured batch SEs.

Final results, mean ± sample SD across five seeds:

| Grid / budget | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid | Coverage ≥1% |
|---|---|---:|---:|---:|---:|
| 3×3 / 50k | vanilla | 0.542 ± 0.001 | 0.679 ± 0.031 | 94.9% | 4.0/9 |
| 3×3 / 50k | paired | 0.058 ± 0.006 | 0.621 ± 0.010 | 94.3% | 9.0/9 |
| 3×3 / 50k | paired_near | 0.944 ± 0.056 | 0.969 ± 0.041 | 36.8% | 0.6/9 |
| 3×3 / 50k | paired_deficit | 0.245 ± 0.422 | 0.702 ± 0.174 | 75.5% | 7.2/9 |
| 3×3 / 50k | vanilla_deficit | 0.454 ± 0.117 | 0.689 ± 0.058 | 94.4% | 4.8/9 |
| 5×5 / 50k | vanilla | 0.718 ± 0.023 | 0.794 ± 0.015 | 73.7% | 7.2/25 |
| 5×5 / 50k | paired | 0.242 ± 0.050 | 0.597 ± 0.034 | 75.8% | 24.4/25 |
| 5×5 / 50k | paired_near | 0.982 ± 0.020 | 0.992 ± 0.015 | 10.5% | 0.6/25 |
| 5×5 / 50k | paired_deficit | 0.214 ± 0.022 | 0.606 ± 0.050 | 78.6% | 25.0/25 |
| 5×5 / 50k | vanilla_deficit | 0.603 ± 0.075 | 0.748 ± 0.045 | 69.6% | 10.2/25 |
| 7×7 / 150k | vanilla | 0.520 ± 0.042 | 0.723 ± 0.026 | 76.3% | 21.0/49 |
| 7×7 / 150k | paired | 0.231 ± 0.045 | 0.647 ± 0.058 | 78.2% | 41.4/49 |
| 7×7 / 150k | paired_near | 0.963 ± 0.016 | 0.977 ± 0.011 | 10.9% | 1.6/49 |
| 7×7 / 150k | paired_deficit | 0.183 ± 0.010 | 0.573 ± 0.030 | 81.7% | 49.0/49 |
| 7×7 / 150k | vanilla_deficit | 0.320 ± 0.045 | 0.681 ± 0.052 | 77.3% | 33.0/49 |

The useful signal comes from emphasizing underrepresented modes, not from the D-only nearby matching tested here. Adaptive sampling improves final mean mode TV for vanilla on every grid and for paired on 5×5/7×7, but destabilizes one paired 3×3 seed. Neither policy is a universal improvement.

Nearby paired performs poorly: final mean mode TV is 0.944 on 3×3, 0.982 on 5×5, 0.963 on 7×7. The corresponding random-paired scores are 0.058, 0.242, 0.231. The matching diagnostics confirm reduced pair distances and the real-real audit is consistent with chance source-role prediction. However, D learns on matched pairs while G uses independent random references; the result may reflect that context mismatch and does not reject nearby pairing in both phases. One-to-one matching also cannot make every reference local after G collapses, because it must retain the full real batch.

On 7×7 at 150k, adaptive paired improves mode TV 0.231→0.183 and fine-grid TV 0.647→0.573. It wins those metrics in 4/5 and 4/5 matched seeds, respectively. Original-threshold coverage improves 41.4→49.0 of 49 modes; all 49 clear that threshold in all five adaptive-paired seeds, versus one original-paired seed. Adaptive vanilla also benefits substantially: mode TV 0.520→0.320, fine-grid TV 0.723→0.681. Thus an appreciable part of the benefit comes from the real-sampling rule itself, not uniquely from joint inputs. Paired remains ahead of the adaptive vanilla control at this endpoint. Both remain above the real-sample fine-grid reference of 0.264.

On 5×5 at 50k, adaptive paired improves mode TV 0.242→0.214, with all 25 modes above the original 1% threshold in all five seeds. But fine-grid TV slightly worsens (0.597→0.606). Adaptive vanilla improves mode TV 0.718→0.603 while remaining far behind paired. On 3×3, adaptive vanilla improves mode allocation (two seeds reach six modes rather than four), without improving mean fine-grid TV. Adaptive paired has four full-coverage seeds and one catastrophic failure: seed 1 has zero valid samples at 50k, raising mean mode TV to 0.245 versus 0.058. Generated coordinates and losses remain finite; the failure is retained, not excluded.

Interpretation: prioritizing missing modes is a promising synthetic-data intervention, with meaningful benefits beyond ordinary paired training on 7×7. It uses known Gaussian geometry and changes D’s real training distribution, so it is not a pure reference-association test or a ready-made image method. The vanilla control uses the same algorithm driven by its own G, not the same realized weights. Paired-deficit also retains uniform G references, introducing a D/G context difference. No mixture strength or EMA tuning was performed, and the seed failure and mixed density rankings matter. Further tests could isolate the phase choice, but none were added to this run.

[Full report, sampling weights, matching diagnostics and all-seed samples](results/grid-reference-comparison-v1/REPORT.md).
All 320 endpoints verified, including 120 exact new checkpoint replays and 200
unchanged baseline sample hashes. G reference/noise/slot/eval and D-noise RNG
states match at all 75 mutually retained checkpoints. Configurations, source
hashes, optimizer budgets, weight normalization/floor and reduced matching cost
verified. 98 tests passed. SciPy's assignment solver is recorded in the lockfile.
Full checkpoints remain locally ignored; source, figures, metrics and small
sample arrays are retained for Git.

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
2. **Further reference interventions:** #20 tests D-only near matching and
   adaptive deficit sampling. Matching in both D and G, explicit far associations,
   and learned-feature matching on images remain untested.
3. **Hyperparameter sensitivity and replication:** G/D learning rates, update
   ratio, capacity, and pair-interaction architecture have not been swept.
   CIFAR conclusions need more seeds before claiming a reliable ranking.

No auxiliary-classification-loss experiment, dedicated mode-recovery test, or
hyperparameter sweep has been completed. #20 adds D-only reference interventions. Experiment #9 now measures the learning curves every 10k through 100k;
a universal “twice as many steps” convergence penalty is still not established. The mechanism behind the toy advantage remains unresolved.

## 21. Unconditional CIFAR class deficit sampling

[Protocol](docs/CIFAR_DEFICIT_PROTOCOL.md) · [Report](results/cifar10-deficit-v1/REPORT.md).
All four seed-0 runs completed 100k updates and all 40 endpoint evaluations.
Original unconditional architectures and D/G batches; only adaptive arms reweight
D real classes using ResNet56 confidence-filtered generated deficits. G references
remain uniform. VGG16-BN is a separate evaluator, not used for sampling.

Final 100k results (lower is better for all columns):

| Method | Held-out FID | KID | VGG class TV | VGG accepted-mass TV, confidence ≥0.9 |
|---|---:|---:|---:|---:|
| Vanilla | 62.23 | 0.04754 | 0.3817 | 0.4585 |
| Vanilla + deficit | 60.23 | 0.04634 | 0.3622 | 0.4515 |
| Paired | 48.79 | 0.03398 | 0.1759 | 0.3117 |
| Paired + deficit | 50.85 | 0.03583 | 0.1590 | 0.3096 |

Deficit improves final raw class TV under both classifiers for both methods,
but the paired quality result is mixed: worse FID/KID, better recall
(0.3316→0.3500), nearly unchanged precision (0.5685→0.5716).
Both paired arms cover all ten classes at the confidence-filtered ≥1%-mass
threshold under both classifiers. Both vanilla arms cover nine under VGG and
eight under ResNet56. Class coverage is not within-class diversity.

The early gains are not monotonic. At 60k, vanilla/vanilla-deficit FIDs were
51.97/48.41, versus 62.23/60.23 at 100k; VGG class TVs worsened from
0.2516/0.1652 to 0.3817/0.3622. Paired/paired-deficit class TVs also worsened
from 0.1353/0.1065 at 60k to 0.1759/0.1590 at 100k. This does not establish
convergence or the cause of late deterioration. It remains a single-seed pilot.

Validation: 104 local tests passed; A10 exact resume, original uniform
trajectory equivalence, and preserved G/noise RNG checks passed.

## 22. G-only grid deficit references

[Protocol](docs/GRID_REFERENCE_G_PROTOCOL.md) · [Report](results/grid-reference-g-comparison-v1/REPORT.md).
All 15 runs completed. D real sampling stays uniform; only G references are
weighted with the existing EMA .99 and 50% uniform floor. Same five seeds,
architectures, losses, batches, and budgets; all historical baselines reused.

Mean final mode TV (lower is better):

| Grid / endpoint | Original paired | D-only deficit | G-only deficit |
|---|---:|---:|---:|
| 3×3 / 50k | 0.0582 | 0.2447 | 0.0945 |
| 5×5 / 50k | 0.2423 | 0.2144 | 0.2513 |
| 7×7 / 150k | 0.2310 | 0.1830 | 0.2348 |

The 3×3 D-only mean includes its previously observed catastrophic seed; no
failures were removed. G-only covers all modes in 5/5 seeds on 3×3, 3/5 on
5×5, and 1/5 on 7×7. Original paired counts are 5/5, 4/5, 1/5; D-only counts
are 4/5, 5/5, 5/5.

On 7×7, G-only mean coverage is 41.0/49 (original 41.4; D-only 49.0).
Fine-density TV is 0.6245 versus 0.6469 original and 0.5729 D-only.
G-only beats original paired on both mode and fine TV in 3/5 matched seeds,
but loses both metrics to D-only in all five. On smaller grids, its mean fine
TV worsens versus original paired (3×3 0.6205→0.7216; 5×5 0.5972→0.6381).

This weighting policy supplies no consistent improvement when moved to G.
It does not establish that every G reference-selection strategy is ineffective,
nor test weighting both phases or selecting references separately for each fake.

Verification: 105 tests passed, 40 new endpoint checkpoint replays matched
exactly, 25 full-run D/noise/slot/evaluation RNG comparisons passed; historical
coarse and fine metrics recomputed from saved samples. Source snapshots retained.
