# Learned flow matching: iterative ablations (#32)

Objective: approach the analytic velocity sampler's distributional fit with a
learned flow, using only sampled real points during training. Oracle-supervised
arms are diagnostics, explicitly excluded from sample-only baseline results.
Original results and checkpoints remain frozen. No GAN retraining.

## Round 1: locate the bottleneck

Eight seed-0 screens, 20k updates, batch 256, primary midpoint128 solver, 4k
fixed evaluation points. All use Adam lr .001, betas (.9,.999), and report raw
and exponential-moving-average weights (decay .999). Variations: raw 2-layer,
raw 4-layer, normalized 3-layer, normalized Fourier 3-layer, preconditioned
3-layer, preconditioned Fourier 3-layer, and two exact-velocity teacher arms.
All hidden layers width128 with SiLU. This is exploratory model selection on
seed 0, not a multi-seed confirmatory comparison.

A separate 100k-point sample-only pilot estimates a scalar standard deviation
s_data. Preconditioning uses s_t²=(1-t)²+t² s_data², provides x/s_t to the
network, and adds the Gaussian velocity [t s_data²-(1-t)]/s_t² times x to its
predicted residual. It does not know grid centers, spacing, labels or Gaussian
component sigma. Fourier features use ten generic frequencies logarithmically
spaced from .5 to 16 cycles per input coordinate, plus time features. The
frequencies were specified before observing screen outcomes.

## Round 2: refine the promising representation

Six fresh seed-0 trials to 50k, 10k evaluation samples. Compare constant lr,
cosine lr (.001 to .0001), batch1024 with cosine, two hidden layers with cosine,
late-biased time sampling with cosine, and width256 with cosine. Other models
are width128, three hidden layers, preconditioned Fourier, batch256. Late time
uses sqrt(U), changing the objective's time weighting; it is labeled separately.
Compute and real-sample budgets vary and must be reported.

## Validation

Choose an architecture using seed-0 development results, then validate on seeds
1–4 as well as the development seed. Compare raw and EMA results transparently,
with the deployment choice fixed before validation. Check solver128 vs256 and
independent evaluation noise. Use the same 10k-sample metrics and finite-sample
reference as the GANs. Exact field supervision never counts as a learned baseline.
Save source snapshots, configuration, RNG/optimizer/model checkpoints, metrics
and sample arrays. No new long training without evidence from short runs.

## Round 3: low-rate refinement

After confirmation, all five selected three-layer checkpoints continue from 50k
to 100k, preserving Adam and random streams. Cosine lr .0001 to .00001 over this
extension; same architecture, batch and objective. Retain 75k and 100k. This is
an adaptive follow-up using the validation findings, so the five seeds should
not be described as untouched model-selection holdouts for this later decision.
Evaluate with the same independent noise used in confirmation and report both
50k and 100k outcomes. Exact sample boundary replays required before training.

A wider (three-layer width256) candidate from round 2 was also confirmed on
seeds1–4 at 50k. It uses the same objective and features. Its extra capacity and
compute are reported explicitly, not treated as architecture-matched to GANs.
