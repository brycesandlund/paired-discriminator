# Training runs

Last updated: 2026-10-08. Running overview of completed research experiments.
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

Representation and optimizer tuning substantially change the 10×10 picture
(#32–33). Tuned flow and both tuned GAN objectives reach all 100 half-target
modes in every confirmation seed at 50k updates. With the same feature and
linear-deficit settings, paired and vanilla have similar mode TV (.070/.067).
The earlier grid advantages therefore depend on the architecture and training
settings tested; they do not establish a general paired advantage. Tuned GANs
still have more within-mode density error than tuned flow on this problem.
The uniform-sampling confirmation (#34) retains about 96% valid mass for both
GANs, but only one of five seeds per method reaches full half-target coverage.
Deficit improves mode balance in every matched seed; uniform has slightly lower
mean raw fine TV. Tuned vanilla-uniform is competitive with paired-uniform.

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
| 23 | Dual-slot D-only grid deficit sampling | dual_slot; seeds 0–4; reuse prior grid arms | 3×3/5×5 to 50k; 7×7 to 150k | Better coverage consistency, no density win: 7×7 full coverage 4/5 vs 2/5 original dual_slot; fine TV worsens 0.595→0.614. Paired D-deficit remains stronger. |
| 24 | Fixed-spacing 10×10 grid | Vanilla, paired, each uniform vs D-deficit; seeds 0–4 | 150k, retained 10k/50k/100k/150k | At 150k all arms remain diffuse. Paired D-deficit has widest lenient coverage (98.2/100) and lowest mean fine TV (0.836); vanilla has lowest mode TV (0.688). |
| 25 | 10×10 exact continuation | Resume all #24 methods/seeds | 150k → 300k; eval every 50k | Paired D-deficit leads both TVs at 300k in every matched seed; mode TV 0.488 vs 0.549 paired / 0.580 vanilla-deficit / 0.619 vanilla. Still far from fitting all modes. |
| 26 | 10×10 exact continuation to 600k | Resume all #25 methods/seeds | 300k → 600k; eval every 50k | Complete: paired D-deficit best mean mode/fine TV (0.290/0.641); 76/100 modes at half-target mass. Still improving through 600k. |
| 27 | 10×10 exact continuation to 1.2M | Resume all #26 methods/seeds | 600k → 1.2M; eval every 50k | Complete: paired D-deficit best mean mode/fine TV (0.209/0.625); 82.8/100 modes at half-target mass. Lenient presence coverage declines 96.4 → 92.8; no seed achieves full half-target coverage. |
| 28 | 10×10 paired D+G deficit | Five fresh seeds; reuse paired uniform and D-only baselines | 0 → 1.2M; eval every 50k | Complete: D+G bias trails D-only at 1.2M: mode TV 0.234 vs 0.209, fine TV 0.648 vs 0.625, half-target coverage 80.4 vs 82.8. |
| 29 | 10×10 stronger D deficit | Five fresh paired seeds; α=.9, p=2; G uniform | 0 → 1.2M; eval every 50k | Complete: stronger bias reaches 91.0 half-target modes vs 82.8; fine TV 0.614 vs 0.625, mode TV 0.214 vs 0.209. |
| 30 | Vanilla flow matching on 10×10 | Five fresh seeds, uniform data | 50k; eval 10k/25k/50k | Complete: at 50k, mode TV .8575, fine TV .9248, valid mass 14.2%, zero half-target modes; 128→256 solver steps has negligible effect. |
| 31 | Flow matching extension | Resume five #30 seeds | 50k → 1.2M | Complete: mode TV .6166, valid mass 38.5%, half-target coverage 24.8/100; solver refinement negligible. |
| 32 | Flow-matching tuning against analytic reference | 31 short jobs; five-seed validation of sample-only models | 20k screens, 50k confirmation, 100k refinement | Complete: tuned flow reaches all 100 half-target modes in every seed; near-analytic mass/density metrics. |
| 33 | GAN feature, optimizer and deficit tuning | 53 short jobs; paired and vanilla, final seeds 0–4 | 30k screens, 50k refinement/confirmation | Complete: unchanged BCE with Fourier G/D features and linear D deficit reaches all 100 half-target modes in every seed for both methods; mode TV .070 paired / .067 vanilla. |
| 34 | Tuned GANs without deficit | Vanilla/paired uniform, seeds 0–4; nine new jobs, reuse paired seed0 and #33 deficit arms | 50k | Complete: uniform mode TV .123 vanilla / .137 paired, versus .067/.070 with linear deficit. About 96% valid mass throughout; uniform full half-target coverage 1/5 seeds per method versus deficit 5/5. |
| 35 | Native-image flow matching | Unconditional MNIST seeds0–4 and CIFAR seed0; existing evaluators | 10k pilot; midpoint64, seed0 midpoint128 check | Complete: MNIST digit TV .098 between vanilla .140 / paired .082, higher acceptance85.9%; CIFAR held-out FID68.99 versus96.90/122.54. Solver sensitivity negligible; no50k extension. |
| 36 | Best image checkpoints: GANs vs flow | Original unconditional vanilla/paired; flow MNIST seeds0–4 + CIFAR seed0 | MNIST through50k; CIFAR through100k | Complete: selected MNIST digit TV .1261 vanilla / .0745 paired / .0633 flow; CIFAR FID49.85 /48.37 /47.02. Paired retains best CIFAR KID and VGG class TV; unequal compute, single-seed CIFAR. |
| 37 | CIFAR optimizer and EMA tuning | Vanilla/paired/flow; uniform unconditional | 16 seed-0 screens, saved EMA, 30k refinement; frozen recipes confirmed on seeds 1–2 | Complete: EMA is the clearest gain. Mean FID 51.38 vanilla / 49.03 paired / 46.62 raw flow / 42.79 secondary flow+EMA. Vanilla extra training regresses; 6.50 GPU-hours. |
| 38 | Paired CIFAR architecture comparisons | Original/wider concat, shared scoring, global context, self-/cross-attention; slot-symmetry probe | Fixed 50k EMA; seed-0 screen then frozen fresh seeds 1–2 | Complete: cross-attention / independent shared scoring mean FID 49.06 / 48.99 vs original 51.68. Cross-attention has highest recall, but worse class balance; no uniform mode-coverage win. 5.23 A10 GPU-hours. |

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

## 23. Dual-slot D-only grid deficit sampling

[Protocol](docs/GRID_DUAL_DEFICIT_PROTOCOL.md) · [Report](results/grid-dual-deficit-comparison-v1/REPORT.md).
All 15 runs completed. Weighted D real points include both members of RR pairs;
G training is unchanged. Same five seeds, EMA .99, 50% uniform floor and budgets.

Mean final results; arrows show original dual_slot → D-deficit dual_slot:

| Grid / endpoint | Mode TV ↓ | Fine-density TV ↓ | Full coverage seeds |
|---|---:|---:|---:|
| 3×3 / 50k | 0.0727 → 0.0647 | 0.6024 → 0.6753 | 5/5 → 5/5 |
| 5×5 / 50k | 0.2105 → 0.2271 | 0.6469 → 0.6631 | 4/5 → 5/5 |
| 7×7 / 150k | 0.2089 → 0.2076 | 0.5951 → 0.6143 | 2/5 → 4/5 |

At 7×7, mean coverage improves 44.8→46.4 of 49 modes. One new seed still
covers only 36 modes. Paired D-deficit remains stronger on mean mode TV
(0.1830), fine-density TV (0.5729), and coverage (49/49 in all five seeds).
Dual-slot D-deficit beats paired D-deficit on mode TV in 2/5 matched seeds,
but loses on fine-density TV in all five. It beats original dual_slot on each
TV metric in only 2/5 matched seeds on 7×7. Coverage gains do not yield a
consistent distribution-fit improvement under this rule and budget.

Validation: 106 tests passed; 40 endpoint checkpoint replays and 25 G/noise
RNG comparisons passed; historical metrics recomputed. The first report attempt
selected the final baseline checkpoint for an intermediate endpoint; this was
corrected to the matching checkpoint and verification rerun successfully.
Training was unaffected. All source/configuration snapshots and seeds retained.

## 24. Fixed-spacing 10×10 grid

[Protocol](docs/GRID10_PROTOCOL.md) · [Report](results/grid10-comparison-v1/REPORT.md).
All 20 runs completed: vanilla/paired, uniform vs D-deficit, five seeds, 150k.
Spacing remains 1.5 and sigma .1; centers extend to ±6.75. Networks, losses,
batches and optimizer are unchanged. Density bounds expand to ±7.75 with .05
cells. Legacy 1% coverage equals a full mode's target mass at 100 modes, so
relative thresholds were specified before results.

Final means across five seeds:

| Method | Mode TV ↓ | Fine TV ↓ | Valid mass | Coverage ≥50% target | Coverage ≥8% target |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.6875 | 0.8420 | 0.3455 | 22.4/100 | 76.8/100 |
| Paired | 0.7223 | 0.8533 | 0.2787 | 7.6/100 | 95.6/100 |
| Vanilla D deficit | 0.7163 | 0.8626 | 0.2863 | 15.6/100 | 81.2/100 |
| Paired D deficit | 0.6932 | 0.8355 | 0.3068 | 12.2/100 | 98.2/100 |

Paired D-deficit improves both TVs over original paired in 4/5 matched seeds.
Vanilla D-deficit improves either TV over vanilla in only 1/5. Against vanilla,
paired D-deficit wins mode TV in 2/5 and fine TV in 3/5; there is no clear overall
winner. No method covers all modes at half-target mass in any seed. The lenient
coverage advantage reflects thin mass spread broadly, not a solved mixture.
Seed-0 scatter plots show substantial bridges and diffuse interiors.

True-mixture calibration (200 draws of 10k): mean mode TV 0.0452 and fine TV
0.3622. All models remain far from that reference. These are budget-limited
results, not demonstrated convergence; larger mode count also changes grid extent.

Validation: 107 tests passed; all 80 retained checkpoints replay exactly;
40 adaptive/control G/noise/slot/evaluation RNG comparisons passed. Source
hashes, configurations and optimizer step counts verified. All seeds retained.

## 25. 10×10 exact continuation to 300k

[Protocol](docs/GRID10_CONTINUATION_PROTOCOL.md) · [Report](results/grid10-300k-comparison-v1/REPORT.md).
All 20 runs resumed exactly from #24, with retained evaluations at 200k, 250k
and 300k. Networks, Adam, RNG streams and deficit EMA restored; no training
hyperparameter changes. Original 150k results preserved.

Final means across five seeds:

| Method | Mode TV ↓ | Fine TV ↓ | Valid mass | Coverage ≥50% target | Coverage ≥8% target |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.6186 | 0.7875 | 0.5269 | 29.2/100 | 73.6/100 |
| Paired | 0.5492 | 0.7653 | 0.4710 | 36.0/100 | 92.2/100 |
| Vanilla D deficit | 0.5803 | 0.7823 | 0.4424 | 36.6/100 | 78.6/100 |
| Paired D deficit | 0.4883 | 0.7151 | 0.5132 | 48.6/100 | 97.8/100 |

Paired D-deficit beats each other arm on both TV measures in all five matched
seeds at 300k. Its mode TV falls 0.6932→0.4883 and half-target coverage rises
12.2→48.6 from 150k. Original paired also overtakes vanilla on mean mode and
fine TV; the 150k ranking was budget dependent.

None achieves full half-target coverage. About half of paired D-deficit's
samples remain outside the acceptance disks. The true-mixture reference is
mode TV 0.0452 / fine TV 0.3622 at 10k samples. Continued improvement through
300k does not establish convergence or a plateau. Lenient presence coverage
alone remains insufficient evidence of a well-fitted mixture.

Validation: 111 tests passed, including four exact continuation tests comparing
all model/Adam/RNG/EMA tensors. All 20 starting sample arrays match #24 exactly;
80 retained endpoint replays, 40 adaptive/control G/noise RNG comparisons, source
hash and optimizer-count checks passed. All seeds retained.

## 26. 10×10 exact continuation to 600k

[Protocol](docs/GRID10_600K_PROTOCOL.md) · [Report](results/grid10-600k-comparison-v1/REPORT.md).
All 20 continuations completed, with evaluations every 50k. Models, Adam, RNG
streams and deficit EMA restored exactly from 300k. No other training changes.
Evaluation overhead measured before launch was ~0.1% of previous loop runtime.

Final means across five seeds:

| Method | Mode TV ↓ | Fine TV ↓ | Valid mass | Coverage ≥50% target | Coverage ≥8% target |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.5928 | 0.7755 | 0.6566 | 34.6/100 | 71.2/100 |
| Paired | 0.4295 | 0.7018 | 0.6368 | 51.8/100 | 91.4/100 |
| Vanilla D deficit | 0.4830 | 0.7384 | 0.5905 | 48.8/100 | 80.0/100 |
| Paired D deficit | 0.2898 | 0.6406 | 0.7278 | 76.0/100 | 96.4/100 |

Paired D-deficit's mode TV improves 0.4883→0.2898 from 300k, and half-target
coverage rises 48.6→76.0. It wins mode TV against every other arm in all five
matched seeds; fine TV wins are 4/5 against paired and 5/5 against both vanilla
arms. No seed achieves full half-target coverage. Mean mode TV continues to fall
0.3252→0.3033→0.2898 at 500k/550k/600k; no demonstrated plateau.

All 140 retained endpoint replays, 70 adaptive/control G/noise RNG comparisons,
source hashes and optimizer-count checks passed. Original boundaries verified.

## 27. 10×10 exact continuation to 1.2M

[Protocol](docs/GRID10_1200K_PROTOCOL.md) · [Report](results/grid10-1200k-comparison-v1/REPORT.md).
All 20 runs completed 600k→1,200,000 updates, with evaluations every 50k.
Networks, Adam, RNG and deficit EMA restored exactly; no other training changes.

Five-seed means (600k → 1.2M):

| Method | Mode TV ↓ | Fine TV ↓ | Coverage ≥50% target |
|---|---:|---:|---:|
| Vanilla | 0.5928 → 0.5612 | 0.7755 → 0.7436 | 34.6 → 39.6 |
| Paired | 0.4295 → 0.3570 | 0.7018 → 0.6548 | 51.8 → 63.0 |
| Vanilla D deficit | 0.4830 → 0.3853 | 0.7384 → 0.6670 | 48.8 → 61.6 |
| Paired D deficit | 0.2898 → 0.2088 | 0.6406 → 0.6248 | 76.0 → 82.8 |

Paired D-deficit remains best on mean mode and fine TV. Its valid mass rises
72.8%→82.8%, but lenient ≥8%-of-target coverage falls 96.4→92.8 modes.
All methods lose some lenient presence coverage despite improved half-target
coverage: mass fitting improves without uniformly preserving mode presence.
No seed reaches full half-target coverage. These endpoints do not establish
convergence, and fine-density TV remains far above the real-sample baseline.

All 260 endpoint replays and 130 adaptive/control RNG comparisons passed,
including source hashes and optimizer-count checks. Completion monitor paused.

## 28. 10×10 paired deficit in both phases

Five fresh paired seeds 0–4 to 1.2M updates; evaluations/checkpoints every 50k.
Both D real examples and G real references use 50% uniform plus 50% normalized
positive accepted-mass deficit, based on the existing EMA (decay .99). Separate
real RNG streams. Same models, losses, batches and optimizer settings as #24–27.
Compare with retained uniform paired and D-only paired baselines; no retraining
of baselines. Status: complete. A smoke check verified two weighted draws per
round with separate RNG streams. Results: `results/grid10-both-1200k-v1`;
automatic report: `results/grid10-both-comparison-v1/REPORT.md`.

Command: `uv run python -m paired_discriminator.grid10_both_run`.

Experiment 28 final means across five seeds:

| Paired sampling | Mode TV ↓ | Fine TV ↓ | Coverage ≥50% target |
|---|---:|---:|---:|
| Uniform | 0.3570 | 0.6548 | 63.0/100 |
| D-only deficit | 0.2088 | 0.6248 | 82.8/100 |
| D+G deficit | 0.2340 | 0.6475 | 80.4/100 |

Adding G-reference bias did not improve final mean performance over D-only bias.
All 120 checkpoint replays and five final noise/slot/eval RNG comparisons passed.
[Report and curves](results/grid10-both-comparison-v1/REPORT.md). Completion check paused.
Future experiments should start with shorter budgets and intermediate comparisons
before extensions, per user feedback on iteration pace.

## 29. Stronger D-only deficit sampling

User-requested fresh 10×10 run to 1.2M, seeds 0–4. D real weights are
0.1/K + 0.9 × squared positive deficit / sum of squared positive deficits.
Uniform fallback if deficits vanish. G references remain uniform. All other
settings match the original D-only experiment, with checkpoints every 50k.
Formula and two-step sampling smoke checks passed. Five local workers.
Monitor intermediate results at common checkpoints against existing D-only runs.

Command: `uv run python -m paired_discriminator.grid10_strong_run`.
Results: `results/grid10-strong-1200k-v1`; report:
`results/grid10-strong-comparison-v1/REPORT.md`. Status: complete.

Experiment 29 final five-seed means at 1.2M:

| D-only bias | Mode TV ↓ | Fine TV ↓ | Valid mass | Half-target coverage | Lenient coverage |
|---|---:|---:|---:|---:|---:|
| Original α=.5, p=1 | 0.2088 | 0.6248 | 82.8% | 82.8/100 | 92.8/100 |
| Stronger α=.9, p=2 | 0.2143 | 0.6143 | 79.2% | 91.0/100 | 99.2/100 |

Stronger bias improves mean coverage and fine-density TV, while original bias
retains slightly lower mode TV and higher valid mass. This is a mixed outcome,
not a uniform improvement; alpha and power changed together. All 120 checkpoint
replays and five final RNG comparisons passed, including uniform G reference RNG.
[Report and plots](results/grid10-strong-comparison-v1/REPORT.md). Monitor paused.

## 30. Vanilla straight-line flow matching

Five fresh seeds 0–4; 50k updates; evaluations at 10k, 25k, 50k.
Independent 2D standard-normal noise and uniform target examples; t uniform
on [0,1], interpolate x_t=(1-t)x_0+t*x_1, regress velocity x_1-x_0 by MSE.
Velocity MLP: two 128-wide SiLU hidden layers with scalar time concatenated.
Batch 256, Adam lr .0002 and betas (.5,.999). No mode labels or deficit sampling.
Midpoint solver 64/128/256 steps (128/256/512 network evaluations), fixed 10k
noise samples for solver comparisons. Report training and sampling time and
real sample consumption; update counts are not equivalent to GAN compute.
Constant-velocity solver check and gradient smoke check passed.

Status: complete. Command: `uv run python -m paired_discriminator.grid10_flow`.
Report: `results/grid10-flow-v1/REPORT.md`.

Experiment 30: final five-seed means at 50k, midpoint 128 steps: mode TV
0.8575, fine TV 0.9248, valid mass 14.25%, half-target coverage 0/100.
128→256 integration steps changes mean mode TV by less than 0.0001; solver
resolution does not explain this poor fit. Seed-0 final samples replay exactly.
Training takes about 24 seconds per seed (five parallel workers); about 39
seconds per seed including evaluations. Each run consumes 12.8M real points.
Sampling requires 256 velocity-network evaluations at the primary setting,
versus a single generator pass for the GANs. At their much larger 1.2M budgets,
original/strong paired D-deficit mode TVs were .2088/.2143. These are different
training and inference budgets, not a matched-compute ranking. This small,
untuned flow model at 50k is a poor fit; no conclusion about flow matching in
general follows. Completion monitor paused.

## 31. Flow matching extension to 1.2M

Resume all five #30 seeds with model, Adam and RNG states restored; replay
50k boundary samples before training. Evaluate at 150k/300k/600k/1.2M with
64/128/256 midpoint steps. Same model, objective and training settings.
At 1.2M, 307.2M real points per run matches D real-data consumption in the
GAN arms; paired GANs additionally consume real G references. Equal updates
and this data budget do not equate compute. Report cumulative training time.
Status: complete. Output: `results/grid10-flow-1200k-v1/REPORT.md`.

Experiment 31 final means, five seeds, midpoint128: mode TV .6166, fine TV
.7753, valid mass 38.45%, half-target coverage 24.8/100. Midpoint256 yields
mode TV .6166 and fine TV .7752: integration resolution is not the explanation.
All five final saved samples replay exactly; all five resume boundaries verified.
Cumulative training time averages about 599 seconds per seed; the extension
finished in about ten minutes with five parallel workers. Each run consumed
307.2M real points. This matches GAN D real draws but not paired G reference
draws or training compute. Sampling uses 256 network evaluations at midpoint128
versus one generator pass in the GANs. Existing 1.2M vanilla / paired / original
paired D-deficit mode TVs: .5612 / .3570 / .2088. This particular flow setup
remains worse at this budget, with no claim about tuned flow matching generally.
[Report and samples](results/grid10-flow-1200k-v1/REPORT.md). Monitor paused.

### Analytic velocity diagnostic after #31

Derived exact E[x1-x0 | x_t] for the known Gaussian mixture and independent
standard-normal source. No model training. Same seed-0 10k source points and
midpoint solver: at 128 steps, mode TV .0461, fine TV .3683, valid mass 98.84%,
and 100/100 half-target modes. 64/256-step results are close. This is near the
finite-sample target reference, demonstrating the path and solver can resolve
the grid when supplied the correct field. The learned seed-0 field has RMSE
.088 at t=.5 versus .785 at t=.75 and .676 at t=.9 on true interpolation
marginals. This locates substantial fitting error but does not separate network
capacity from optimization or regression-target noise. Oracle uses known mixture
parameters and is not a learned baseline.
[Derivation, metrics and plot](results/grid10-flow-oracle-v1/REPORT.md).

## 32. Flow matching tuning against the analytic reference

[Protocol](docs/FLOW_ABLATIONS_PROTOCOL.md) · [Report](results/flow-ablation-comparison-v1/REPORT.md).

31 short jobs across screening, confirmation, and refinement. Two oracle-teacher trials are diagnostics only. Final learned models train on ordinary independent noise/real pairs with velocity MSE; no mixture parameters or labels. The pilot estimates only data scale. Fixed generic Fourier features, Gaussian scale/residual preconditioning, Adam (.9,.999), and cosine lr .001→.0001 over 50k make the major improvement. Batch256. Both two- and three-hidden-layer networks work.

Five-seed fresh-noise means at midpoint128:

| Model | Mode TV | Fine TV | Valid mass | Half-target coverage |
|---|---:|---:|---:|---:|
| Analytic oracle | 0.0470 | 0.3642 | 98.89% | 100.0/100 |
| Tuned 2×128, 50k | 0.0585 | 0.3725 | 97.40% | 100.0/100 |
| Tuned 3×128, 50k | 0.0553 | 0.3697 | 98.01% | 100.0/100 |
| Tuned 3×128, 100k | 0.0525 | 0.3686 | 98.05% | 100.0/100 |
| Tuned 3×256, 50k | 0.0519 | 0.3698 | 98.68% | 100.0/100 |

All tuned models cover 100/100 half-target modes in every seed. Raw weights fixed before confirmation. Seed0 development, seeds1–4 initial confirmation; follow-up decisions were adaptive. Midpoint256 barely changes results. Original #30–31 flow results were an under-tuned setup and do not represent a competitive flow baseline.

20 final checkpoint/sample replays, all source hashes, optimizer steps, and five refinement resume boundaries passed; three focused tests passed. No claim of exact field equality everywhere. Results are not matched-parameter or matched-compute GAN comparisons. Checkpoints, all trial outcomes and plots preserved; no commits made.

## 33. GAN features, optimizer and deficit sampling

[Protocol](docs/GAN_ABLATIONS_PROTOCOL.md) · [Report and samples](results/gan-ablation-comparison-v1/REPORT.md).

53 short jobs: 35 seed-0 screens across four rounds, 13 initial confirmation
jobs, and five matched vanilla linear-deficit follow-ups. Original BCE D and
non-saturating BCE G objectives throughout; no reconstruction loss. Raw G is
the primary comparison, fixed before confirmation. Final evaluation uses 10k
fresh latent samples per seed; old checkpoints are also re-evaluated. Seed0
participated in development, and later choices were adaptive; these five-seed
summaries include development seed0 rather than five untouched test seeds.

The successful G uses a 2D normal latent, generic sine/cosine features at ten
log-spaced frequencies .5–16, two width128 LeakyReLU hidden layers, and output
`s_data * (z + MLP(features(z)))`. The shared 100k-real-point pilot estimates
only scale. D uses normalized coordinates plus Fourier features introduced
from low to high over the first 25k updates, also two width128 hidden layers.
Neither feature map contains mode centers or grid spacing. The deficit sampler
still uses known mode identities, as in earlier grid experiments.

Both Adam optimizers retain betas (.5,.999), with lr .001 for 25k updates,
then cosine decay to .0001 at 50k. D batch256 real/256 fake; G batch256 fake,
with 256 uniform real references for paired. Linear deficit uses alpha=.9,
power=1, mass EMA decay=.99. The squared/fast alternative uses alpha=.9,
power=2, mass EMA decay=.9. Optional generator-weight EMA decay=.999 is
reported separately for every seed, never selected per seed.

Fresh-noise five-seed means, raw outputs:

| Model | Updates | Mode TV ↓ | Fine TV ↓ | Valid mass | Half-target coverage |
|---|---:|---:|---:|---:|---:|
| Original vanilla | 1.2M | .5628 | .7445 | 79.73% | 39.4/100 |
| Original paired, strong D deficit | 1.2M | .2121 | .6148 | 79.34% | 91.4/100 |
| Tuned paired, squared deficit / fast estimator | 50k | .0889 | .4130 | 95.71% | 99.0/100 |
| Tuned paired, linear deficit | 50k | .0702 | .4260 | 96.61% | 100.0/100 |
| Tuned vanilla, squared deficit / fast estimator | 50k | .0707 | .4208 | 96.32% | 100.0/100 |
| Tuned vanilla, linear deficit | 50k | .0674 | .4265 | 96.41% | 100.0/100 |
| Real mixture reference | — | .0432 | .3623 | 98.91% | 100.0/100 |

Both linear-deficit methods reach all 100 half-target modes in all five seeds.
The squared/fast paired variant does so in only two of five, despite better
mean fine TV. Generator EMA improves fine TV to .3894 paired-linear and .3932
vanilla-linear; real finite-sample fine TV is .3623. All-mode coverage does
not imply perfect balance or correct within-mode density.

D features or broader G initialization alone did not solve coverage in the
short screens. Combining G Fourier features with the residual parameterization
and gradual D features produced the large observed gain. Sampler and late
learning-rate adjustments then improved the result. The gains transfer to
vanilla: this experiment does not demonstrate a paired-only benefit.

Final models train in about 55–64 seconds per seed under concurrent workloads.
Each consumes 12.8M D real draws; paired additionally uses 12.8M G references,
plus the shared scale pilot. Vanilla draws unused G reference batches only to
keep RNG streams aligned. G has 22,274 parameters; D has 27,521 paired or
22,145 vanilla. Inference is one G pass. These are different architectures
from the original runs; 24× fewer updates does not mean 24× less compute.

40 exact raw/EMA checkpoint/sample replays, five matched paired/vanilla RNG
checks, frozen-source hashes and optimizer step counts passed. Three focused
tests passed, including a two-update bitwise regression against the original
trainer. All trials and source snapshots retained; no commits made.

## 34. Tuned GANs with uniform D sampling

[Protocol](docs/GAN_UNIFORM_PROTOCOL.md) · [Report and samples](results/gan-uniform-comparison-v1/REPORT.md).

Freeze #33's successful feature architecture and optimizer schedule, and set
alpha=0 for uniform D real sampling. Paired G references remain uniform.
Train vanilla seeds0–4 and paired seeds1–4, reusing the exactly matched paired
seed0 screen. Nine new jobs complete in 110 seconds elapsed with up to eight
workers; all ten uniform trajectories reach 50k updates. All linear-deficit
checkpoints are reused from #33. No uniform-specific hyperparameter search and
no loss changes. Run: `uv run python -m paired_discriminator.gan_uniform`;
report: `uv run python -m paired_discriminator.gan_uniform_report`.

Raw generators, five-seed means on the same 10k evaluation-noise samples as
#33, independent of training and the earlier screening noise:

| Method | Mode TV ↓ | Fine TV ↓ | Valid mass | Half-target modes | Full coverage seeds |
|---|---:|---:|---:|---:|---:|
| Vanilla uniform | .1228 | .4071 | 96.43% | 96.6/100 | 1/5 |
| Paired uniform | .1373 | .4187 | 96.34% | 95.4/100 | 1/5 |
| Vanilla linear deficit | .0674 | .4265 | 96.41% | 100/100 | 5/5 |
| Paired linear deficit | .0702 | .4260 | 96.61% | 100/100 | 5/5 |

The feature-based improvement survives uniform sampling. Linear deficit lowers
mode TV in every matched seed for both objectives, while uniform raw outputs
have slightly lower mean fine TV. Full half-target coverage is not the same as
merely detecting a mode: each mode must contain at least .5% of all samples
within radius .3. Uniform paired seed2 has the weakest half-target coverage
(86/100). There is no paired advantage in these five-seed means.

Generator EMA is a separately reported option. Its mode/fine TVs are
.1199/.3884 vanilla-uniform and .1340/.3902 paired-uniform. Coverage counts
remain unchanged on final evaluation. Means include development seed0.

All arms use the same G, updates, nominal batches, optimizer schedule and shared
100k-point scale pilot. Each uses 12.8M D real draws; paired additionally uses
12.8M G references and a larger D. Equal updates do not equate compute. With
alpha=0, the trainer's unused deficit estimate cannot affect sampling or losses.

40 exact final raw/EMA checkpoint replays across all four arms, 20 matched RNG
comparison groups, source hashes, config/spec matches, optimizer counts and
completion metadata passed. Training implementation unchanged from #33.
Reports, sources and samples retained; no commits made.

## 35. Unconditional image flow matching on MNIST and CIFAR

[Protocol](docs/IMAGE_FLOW_PROTOCOL.md) · [Report and plots](results/image-flow-v1/REPORT.md).
Completed initial10k pilot: five MNIST seeds and
one CIFAR seed. Native-resolution U-Net with widths32/64/128, GroupNorm/SiLU,
sinusoidal time embedding, and straight-line independent-pair velocity MSE.
1,168,929 parameters on MNIST; 1,170,083 on CIFAR. Adam .0002, betas(.9,.999),
500-step warmup, batch128, uniform real sampling. No labels or deficit sampling.
Raw weights primary; EMA retained but not used for selection.

Full evaluation at10k uses the existing frozen MNIST classifier and CIFAR
classifiers/Inception metrics. Midpoint64 (128 network evaluations per image),
with a full midpoint128 seed0 sensitivity check on each dataset. Retain
checkpoints and midpoint16 previews every1k. Evaluation frequency differs from
the original MNIST every1k full-metric schedule to limit ODE sampling cost.
Equal updates/real draws do not equate compute or inference cost.

Four local tests passed: interpolation/velocity target, midpoint integration,
native shapes and time gradients, and exact CPU resume with sampling neutrality.
CUDA resume/evaluation-neutrality preflight passed for both datasets, including
raw/EMA weights, optimizer state, RNG states and losses. All six runs and eight
full evaluations finished. Source/reference/classifier checks and first-batch
float sample replays passed. Status: complete. Results: `results/image-flow-v1`.
Run: `uv run --extra cifar modal run --detach modal_image_flow.py::main --run-id image-flow-v1 --steps 10000`.
No commits or unrequested extensions.

MNIST10k, five-seed means (raw weights, midpoint64):

| Method | Digit TV ↓ | Covered digits | Confidence acceptance | Confidence-augmented TV ↓ |
|---|---:|---:|---:|---:|
| Original vanilla | .1401 | 10/10 | 76.30% | .2621 |
| Original paired | .0815 | 10/10 | 77.30% | .2323 |
| Flow matching | .0980 | 10/10 | 85.89% | .1826 |

Flow has all-ten-digit raw and confidence-filtered coverage in every seed.
Its digit balance is between the GAN means, while acceptance/augmented TV
improve. This10k checkpoint does not yet test the late instability seen at50k.
Mean flow digit TV SD is .0189 across seeds. Classifier confidence remains an
uncalibrated proxy and does not establish within-digit diversity or fidelity.

CIFAR10k, seed0, matched official held-out test references:

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | ResNet class TV ↓ | VGG class TV ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original vanilla | 96.90 | .08597 | .710 | .040 | .4265 | .3523 |
| Original paired | 122.54 | .10853 | .627 | .007 | .4874 | .4150 |
| Flow matching | 68.99 | .06025 | .641 | .140 | .2358 | .1984 |

Flow improves FID/KID, feature recall and raw class balance at this endpoint,
with lower precision than vanilla. All ten classes exceed1% raw and accepted
mass under both classifiers. Evaluator agreement is only49.67%, and confidence
acceptance is54.21% ResNet /66.50% VGG, so category claims remain tentative.
One seed and different architectures/compute do not establish a general ranking.

Solver64→128 changes MNIST seed0 digit TV .1177→.1174 and CIFAR FID
68.99→69.01, so these results are not materially solver-resolution limited.
Mean MNIST training time is274.6 seconds/seed; CIFAR332.7 seconds, A10-class GPUs.
Primary10k-sample generation costs66–79 seconds on MNIST and92 seconds on CIFAR.
Each generated image requires128 network evaluations versus one GAN G pass.
Each training run consumes1.28M real draws; paired GAN additionally consumes
real G references. Results compare update/data endpoints, not equal compute.

An export-only ZIP timestamp issue was repaired; no training/evaluation reruns
were needed. Four local tests and both CUDA exact-resume checks passed.

## 36. Best image checkpoints: vanilla, paired, flow matching

Completed all five MNIST flow runs through50k and CIFAR seed0 through100k, with
10k-interval evaluations. Original GAN trajectories reused; no deficit sampling,
conditioning, architecture changes or GAN retraining. Original10k flow results preserved.

[Full report](results/image-best-checkpoints-v1/REPORT.md) ·
[Samples](results/image-best-checkpoints-v1/comparison_samples.png) ·
[Class proportions](results/image-best-checkpoints-v1/class_mass.png) ·
[Individual-seed heatmaps](results/image-best-checkpoints-v1/mnist_seed_mass.png) ·
[Verification](results/image-best-checkpoints-v1/verification.json)

| Method | MNIST selected step | Digit TV, five-seed mean ± SD | CIFAR selected step | Held-out FID |
|---|---:|---:|---:|---:|
| Vanilla GAN | 5k | .1261 ± .0100 | 50k | 49.85 |
| Paired GAN | 8k | .0745 ± .0062 | 80k | 48.37 |
| Flow matching | 40k | .0633 ± .0123 | 80k | 47.02 |

MNIST selects one shared step per method by mean individual-seed digit TV.
CIFAR selects by held-out FID. Other primary metrics use those same checkpoints.
Flow MNIST acceptance is90.1% versus72.2% vanilla/76.0% paired at their selected
steps; augmented TV .1304 versus.2824/.2404. Selecting each method by augmented
TV instead gives.1253 flow at50k versus.2589 vanilla at9k/.2057 paired at35k.
Common10k-interval selection preserves the digit-TV ranking. All selected MNIST
runs retain ten digits above1% mass. Class balance does not establish within-class diversity.

CIFAR is mixed: flow has lowest FID and highest precision (.617), but paired
has lowest KID (.03387 versus flow.03786/vanilla.03705) and lowest VGG class TV
(.1673 versus flow.1915/vanilla.2163). Recall is similar (~.338–.340).
CIFAR has one seed and retrospective checkpoint selection; the reference is not
an untouched final test. Flow uses128 network evaluations per sample versus one
GAN generator pass, so this is not equal compute.

All35 primary flow evaluations pass saved first-batch replay checks and frozen
classifier/reference identity checks. Doubling integration steps at selected
seed0 checkpoints changes MNIST TV .07167→.07177 and CIFAR FID47.018→46.891.
Cumulative training: MNIST23.1–23.6min per seed; CIFAR50.7min, excluding evaluation.
Each MNIST flow seed consumes6.4M real draws; CIFAR12.8M through100k.

A10/A10G transitions are recorded in hardware_migration.json after saved10k
first-batch replay with maximum error below.001. Model/optimizer/RNG state was
restored unchanged; cross-GPU continuation is not claimed bitwise identical.
The completion monitor is paused. No commits made.

## 37. CIFAR optimizer and EMA tuning

Complete. Three rounds: screen on seed 0, refine promising settings, then freeze
recipes and confirm on fresh training seeds 1 and 2 with new evaluation noise.
Uniform unconditional CIFAR throughout; original architectures, BCE / velocity
MSE, no deficit sampling or classifier guidance. No architecture feature sweep.

[Full report](results/cifar-tune-v1/REPORT.md) ·
[Samples](results/cifar-tune-v1/confirmation_samples.png) ·
[Class proportions](results/cifar-tune-v1/confirmation_class_mass.png) ·
[All candidates](results/cifar-tune-v1/ALL_CANDIDATES.md) ·
[Verification](results/cifar-tune-v1/final_verification.json)

| Frozen recipe | Total updates | FID, new-seed mean ± SD | VGG class TV | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Vanilla, original Adam + EMA | 60k | 51.38 ± 1.11 | .2384 | .5890 | .3516 |
| Paired, original Adam + EMA | 70k | 49.03 ± .40 | .1598 | .5688 | .3428 |
| Flow, original Adam, raw (primary) | 110k | 46.62 ± 4.08 | .1661 | .6197 | .3544 |
| Flow, quarter LR + EMA (frozen secondary) | 110k | 42.79 ± .03 | .1257 | .6278 | .3743 |

Scores average individual seeds, never pooled distributions. All recipes and
endpoints were fixed before confirmation. Both flow variants remain visible;
we did not switch the primary to the better-looking confirmation result.
Two seeds provide limited evidence. Train-reference FID drove tuning; held-out
scores are descriptive, with shared reference history rather than an untouched test.

**What improved:** EMA is the clearest repeatable intervention. The secondary
flow recipe reaches FID 42.78 / 42.81; original-rate flow EMA at the same
checkpoints is similarly strong (42.57 / 42.86), so lowering the learning rate
has no convincing independent benefit. Primary raw flow is variable (49.50 /
43.74); its seed-0 FID of 42.11 did not reproduce consistently. Further training
improves both paired and flow EMA relative to their earlier EMA bases.

Paired beats vanilla on FID and category balance in both confirmation seeds,
but has slightly lower feature recall. Vanilla's prescribed extra 10k updates
worsen FID in both seeds: earlier 50k EMA bases average 49.39, versus 51.38 at
the frozen 60k endpoint. That narrows the broader paired-versus-vanilla FID
claim; it is not evidence of a large universal quality advantage. Neither class
balance nor feature recall establishes within-class diversity.

**Search and control protocol:** three saved flow EMA evaluations at 50k/80k/100k;
six GAN settings per method (original Adam 2e-4, half both LRs, half D only, half
G only, cosine decay, lazy R1); four flow settings (2e-4, 1e-4, 5e-5, cosine).
Screen 10k branch updates, then refine winners and controls through 30k. All
branches start from matched model weights (GAN 50k / flow 80k), resetting Adam
and sampling RNG identically, including controls. They are warm starts, not
exact continuations. No tested GAN LR or R1 setting beat the original optimizer.

EMA decay .999 includes floating BatchNorm buffers; integer buffers copied.
Cosine reaches 10% of the initial LR at 30k branch updates. R1 gamma 1 is applied
every 16 D updates with interval scaling; paired penalizes both input slots.
Seed-0 selected FIDs were 49.58 vanilla EMA, 48.23 paired EMA, 42.11 raw flow,
and 43.33 secondary flow EMA; these are discovery results, separate from the table.

Confirmation trains six fresh bases, then eight branches. Frozen choices:
vanilla +10k / EMA; paired +20k / EMA; flow +30k original LR / raw as primary,
and +30k quarter LR / EMA as secondary. Evaluation uses 10k samples with new
noise seed 99274 and the same classifiers / reference images as earlier work.
Source and recipe details are saved in `results/cifar-tune-v1`.

**Compute:** 4.94 GPU-hours training + 1.56 evaluation = **6.50 A10 GPU-hours**,
using at most four workers. This excludes startup, checkpoint / volume I/O,
tiny verification runs and earlier reused parents; no failed training observed.
Fresh recipe training takes about 16 min for vanilla, 14–16 min for paired,
and 57–62 min for flow per seed. Total real draws are 7.68M / 17.92M / 14.08M,
respectively (paired also draws references during G updates). This is unequal
compute. GAN inference remains one G forward; flow uses 64 midpoint steps /
128 velocity evaluations. EMA adds no deployment network passes.

All 30 training endpoints completed, all 79 evaluations passed saved first-batch
replay and source / classifier / reference checks. Dataset and all three trainer
source hashes match across runs. Three CUDA split-resume checks and five local
tests passed. Sample and class-mass plots were inspected. The monitor is paused;
no commits made.

Reproduce collection with `python -m paired_discriminator.cifar_tune_report --collect`,
then final reporting with `python -m paired_discriminator.cifar_tune_finalize`.
Launcher: `modal_cifar_tune.py`; remote volume `paired-discriminator-cifar-tuning`.
Logs: `/tmp/cifar-tune-v1.log` and `/tmp/cifar-tune-confirm.log`.

## 38. Paired CIFAR architecture comparisons

Complete. Improving the comparison architecture helps some quality metrics, but
**cross-attention does not clearly outperform simpler shared independent scoring
on FID, and neither improves original paired class balance**. Cross-attention
has higher feature recall and lower KID, with more training cost. These results
do not establish that richer pair comparisons solve mode collapse.

[Full report](results/cifar-pair-arch-v1/REPORT.md),
[fixed confirmation samples](results/cifar-pair-arch-v1/confirmation_samples.png),
[class mass by seed](results/cifar-pair-arch-v1/confirmation_class_mass.png),
[verification](results/cifar-pair-arch-v1/verification_report.json).

**Fresh confirmation seeds 1 and 2, fixed 50k EMA:** means of individual-seed
metrics; seed 0 is excluded. FID uncertainty is sample standard deviation across
these two seeds, not a confidence interval. No scoring of pooled distributions.

| Architecture | Train FID ↓ | Test FID ↓ (mean ± SD) | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ | Training min/seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original concat | 51.95 | 51.68 ± 1.95 | .03558 | .555 | .321 | **.148** | **.204** | 11.33 |
| Shared independent scores | 49.25 | **48.99 ± .34** | .03690 | .574 | .354 | .180 | .223 | 14.77 |
| Self-attention control | 53.68 | 53.69 ± .81 | .03884 | **.595** | .376 | .289 | .314 | 20.62 |
| Cross-attention | **49.12** | 49.06 ± 1.71 | **.03371** | .549 | **.385** | .231 | .260 | 21.28 |
| Antisymmetric concat (exploratory) | 50.91 | 50.68 ± 2.87 | .03513 | .550 | .331 | .180 | .249 | 15.05 |

Cross-attention essentially ties original paired on seed 1 (FID 50.27 vs 50.31),
but improves strongly on seed 2 (47.85 vs 53.06). It improves recall in both while
worsening both classifiers' class TV. VGG assigns only 2.1–2.4% to cars and about
20% to frogs, versus original paired's 3.1–3.3% and 14.4–15.6%; uniform target
mass is 10%. These are predicted labels, not ground-truth generated categories.

Independent shared scoring improves FID, precision and recall in both fresh seeds,
with less FID variation here. Its mean FID is almost identical to cross-attention,
and it has better precision/class balance but lower recall and worse KID.
Its mean KID is slightly worse than original paired, so it is not an across-metric
win either. Cross-attention beats the matched self-attention control's FID in both
seeds. The symmetry probe is mixed: FID 52.71/48.65 versus original 50.31/53.06,
worse/better recall respectively, and worse class TV in both. It also worsened
seed-0 FID (52.98 vs 51.56). Antisymmetry alone did not reliably improve training.

EMA improves FID versus raw weights at every confirmation endpoint; EMA was fixed
before these runs, not selected afterward. Raw/EMA per-seed results are in the
report. Class TV does not always improve with EMA. Fixed sample grids remain
small and often ambiguous; they do not establish dramatic visual gains or
within-class coverage. Two fresh seeds and previously used data references make
this exploratory evidence, not untouched validation.

**Architecture controls and protocol:**

| Arm | Comparison mechanism | D parameters |
|---|---|---:|
| `concat` | Original six-channel CNN, width 64 | 666,049 |
| `concat_wide` | Same architecture, width 66; capacity control | 707,983 |
| `shared_difference` | Shared image CNN, score(A) − score(B); relativistic control | 662,977 |
| `global_context` | Features conditioned on the other image's pooled features | 712,770 |
| `self_attention` | Attention within each image, then score difference | 712,770 |
| `cross_attention` | Each image's features query the other image's features | 712,770 |
| `symmetric_concat` | 0.5 × (D(A,B) − D(B,A)); targeted follow-up | 666,049 |

Shared arms have identical initial CNN tensors. Self-/cross-attention have exactly
the same parameter initialization; only the source of keys/values changes.
Global conditioning has the same parameter count. Attention acts at 8×8 resolution,
64 tokens/image, four heads, 16 query/key and 32 value dimensions per head. The
learned residual scale starts at .1. LayerNorm acts within each token. Shared arms
subtract two scores, enforcing D(B,A) = −D(A,B); independent scoring and
self-attention remain additive unary controls.

G is the original width-64, latent-128 model (1,183,619 parameters), initialized
identically across architectures within each seed. All arms use BCE, Adam 2e-4 /
(.5,.999), 128 pairs for D and 128 generated samples for G, one D and one G update,
uniform unconditional real sampling, no deficit or auxiliary reconstruction.
EMA decay .999 includes the same floating-buffer policy as #37. Training sources
remained frozen throughout active jobs; the symmetry module preserves the original
train/diagnostic loop verbatim with explicit parity tests and its own source hash.

Six seed-0 arms trained fresh to 50k with raw/EMA evaluations at 10k/20k/50k.
The predeclared primary comparison was **50k EMA**, selected by TRAIN-reference
FID. Cross-attention won (train 48.17 / test 48.08), versus original test 51.56,
shared independent 49.19, wider concat 50.33, global context 52.58 and self-attention
52.80. Cross-attention seed-0 recall was .396 vs original .302, but VGG TV .229 vs
.148. Earlier endpoints and raw weights were descriptive.

The frozen confirmation compared cross-attention, original concat, shared scoring
and self-attention on fresh seeds 1 and 2. A single targeted symmetry probe used
seeds 0, 1 and 2, also at 50k EMA. All choices, source hashes and specs were saved
before launch in `confirmation_decision.json` and `confirmation_specs.json`.
Noise seed 99374 was used for seed 0 and new noise 99474 for fresh seeds. No
confirmation checkpoint/config/weight selection or further architecture sweep.

**Interaction diagnostics:** hold each fake fixed and rotate real references;
measure raw-logit gradient-direction cosine before the BCE scalar, plus a
four-pair interaction residual. Shared scoring/self-attention have cosine 1 and
residual near zero. Cross-attention has cosine .949/.903 and residual .900/.994
on fresh seeds. Original concat already strongly uses the reference (cosine
.107/.807), so the mechanism is not simply beginning to use the second input.
Original slot-swap error was 25.54/10.77; the symmetry probe forces zero but yields
mixed outcomes. These use RAW G/D and do not prove useful coverage of EMA samples.

**Compute and verification:** 4.69 GPU-hours training + .54 evaluation = **5.23 A10
GPU-hours**, under the initial approximately six-hour bound, at most four workers.
Excludes startup, checkpoint/volume I/O, tiny CUDA checks, untimed diagnostics and coordinator CPU time.
Cross-attention training costs 1.88× original concat and 1.44× shared scoring.
All methods draw 12.8M real images/references over 50k updates. G is unchanged,
so inference is still one identical generator forward; D changes add no deployment
passes. This is an equal-update/data comparison, not equal training compute.

All **17 endpoints / 58 evaluations** completed. Seven CUDA exact-resume checks
passed, including original concat equivalence; 15 local tests passed. Saved
first-batch replays, dataset/classifier/reference/evaluation identities, frozen
source sets, specs/dispatch and raw/EMA checkpoint identities all passed. Samples,
trajectories and class-mass plots were inspected. No training failures observed.
The heartbeat is paused; no commits made.

One orchestration deviation: Modal preempted the confirmation coordinator after
dispatch; its restart hit the duplicate-dispatch guard. The original 11 training
calls continued. A CPU-only recovery collector rejoined those exact IDs, without
redispatching training, and completed with no job errors. Details are retained in
`confirmation_recovery.json` and `confirmation_recovery_launch.json`.

Source: `src/paired_discriminator/cifar_pair_arch.py` and `cifar_pair_symmetry.py`;
launcher `modal_cifar_pair_arch.py`; volume `paired-discriminator-cifar-pair-architecture`.
Collection: `uv run --extra cifar modal run modal_cifar_pair_arch.py::collect`;
report: `uv run python -m paired_discriminator.cifar_pair_arch_report`.
Initial frozen evidence remains in `results/cifar-pair-arch-v1/screen_REPORT.md`.
Screen coordinator/app: `fc-01M4GRD25SMYWKF1XMCVYJD6E0` / `ap-kunuKA7EKzdKJCDrlmokLj`.
Original confirmation: `fc-01M4GVQXTTHG4HV16YGKFE4Z3Z` / `ap-X5HAwfYaf6AxXxsTfPYjBk`.
Recovery: `fc-01M4GW0V4BT0MXYKA3219CF49B` / `ap-mFczMencQdpR51BbOeDtBg`.
Logs: `/tmp/cifar-pair-arch-v1.log`, `/tmp/cifar-pair-arch-confirm.log`,
`/tmp/cifar-pair-arch-recover.log`. No commits.
