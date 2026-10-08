# GAN feature, optimizer and sampler tuning (#33)

User-authorized adaptive search on the fixed-spacing 10×10 grid. Keep original
BCE discriminator and non-saturating generator objectives. No reconstruction,
transport matching, oracle gradients, or auxiliary generator losses. Start with
paired D-only deficit sampling; keep uniform G references. All prior artifacts
remain frozen. Save configurations, source snapshots and raw/EMA outcomes.

Round 1: ten seed-0 screens, 30k updates, batch256, 4k fixed evaluation samples.
Compare original strong D-deficit control; lr .001; lr .001 with Adam beta1=.9;
D coordinate normalization; D Fourier features; Fourier plus lr .001; Fourier
plus beta1=0,beta2=.99; progressive Fourier-band introduction; and Fourier
with weak (.5,1) or uniform D sampling. Unless overridden, alpha=.9,p=2,
Adam (.5,.999), lr=.0002, two hidden layers width128 with LeakyReLU(.2).

Features are per-coordinate sin/cos at ten generic logarithmic frequencies
.5–16 after normalization by an independent 100k-point pilot standard deviation.
No component centers or spacing in features. Annealing introduces bands from low
to high over the first half of training. Generator architecture/initialization
matches the old experiments in this first round. Optional generator EMA .999 is
reported alongside raw weights; never silently select the better one per seed.

Selection on seed0, then confirm promising configurations across more seeds and
independent noise. Compare the tuned architecture under paired and vanilla
losses when practical; improvements shared by both should not be credited to
pairing. Deficit sampling retains known mode labels as in prior experiments.
Use shared 10k-sample evaluations for final comparisons; 4k screening metrics
have different finite-sample floors. Report architecture, sample and runtime
budgets rather than implying equal compute for equal update counts.

## Round 2: generator spread and feature scheduling

Eight seed-0 30k trials. Add a deterministic skip from the first two existing
latent coordinates: G(z)=s_data*(z[:2]+MLP(z)), retaining all original MLP
parameters and latent dimensions. This broadens initialization without target
mode labels, loss additions or assigned reconstruction targets. Compare raw D,
Fourier D, gradual Fourier features, separate G/D learning rates, and weak or
uniform real sampling. All parameterization changes are separate from the
first round where G remained identical to the original baseline.

## Round 3: generator features

After broader initialization alone failed to recover coverage, compare generic
Fourier features of the first two latent coordinates and latent dimension2
versus16. Keep two 128-wide LeakyReLU hidden layers. Test residual versus plain
outputs, raw versus gradually introduced Fourier D features, cosine schedule,
D SiLU and Adam beta1=0. No coordinates from real examples are passed into G;
this is still unconditional generation. Different latent dimensions do not share
pointwise identical generated samples; evaluations use the same seeds/protocol.

## Round 4: sampler and late optimizer schedule

Nine seed-0 screens to 50k with 10k evaluation samples. Use the successful
2D latent Fourier residual G plus gradual Fourier D. Compare constant lr with
late cosine (.001 held first half, then to .0001); uniform real sampling;
(alpha,p)=(.5,1),(.9,1),(.5,2),(.9,2); mass EMA .9/.999 versus .99; and vanilla
BCE under otherwise identical strong-sampler feature settings. Raw generator
is the primary final comparison: EMA sample quality is retained separately,
not selected per seed based on whichever metric looks favorable.

## Confirmation

Before seeing new seeds, choose paired alpha=.9,p=2,mass-decay=.9 and paired
alpha=.9,p=1,mass-decay=.99 from round4, with late cosine and the same feature
architecture. Confirm each on seeds1–4 (retain development seed0 separately).
Run vanilla with exactly the first setting on seeds0–4. Uniform G references
remain available only where the objective uses them; vanilla has no extra G
reference dependence. Raw is the primary output; report EMA as a separate,
fixed alternative for every seed. Final independent evaluation uses 10k latent
samples with seed 932000+seed, not the development noise. Preserve sample replay,
optimizer and RNG consistency checks. No inference-time mode snapping or
postprocessing: every plotted sample is a direct generator output.

## Matched vanilla follow-up

After confirmation favored linear deficit for reliable paired coverage, run
vanilla with exactly that setting on seeds0–4. This is an adaptive follow-up,
not a configuration selected before examining all confirmation outcomes.
Use the same final fresh-noise evaluation, raw-primary convention, and report
all seeds. Check paired/vanilla noise, uniform G-reference and slot RNG states
for parity; only paired actually uses the G real references in its objective.
