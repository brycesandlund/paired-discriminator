# Grid reference intervention

Prespecified first pass: affect **D only**, keeping G's reference sampling,
batch, optimizer frequency and loss unchanged. Use original D256/G256 settings,
not the doubled-D arm. Existing random paired and vanilla results are controls;
RSGAN/PacGAN2/two-output can be shown as background baselines.

45 new runs: paired-near, paired-deficit, vanilla-deficit × seeds 0–4 ×
3×3/5×5/7×7. Train the smaller grids to 50k and 7×7 to 150k, retaining
10k/50k and additionally 100k/150k for 7×7. Original spacing 1.5, sigma 0.1,
G/D architectures, learning rates, 256 real and fake points per D update,
256 fresh generated points per G update. Eight CPU workers, one torch thread each.

## Nearby references

Draw the same uniform real batch and generated batch as in original paired.
Use minimum-total-squared-Euclidean-cost bipartite matching to permute the real
batch into the original fake order. Preserve every point exactly once, then
randomize A/B slots with the original random stream. This biases association,
not either marginal. It can only reduce squared pairing cost relative to the
original assignment. It is a batch-constrained nearest pairing, not independent
nearest-neighbor retrieval.

Symmetric costs and unique optimal assignments make the procedure equivariant
to swapping sources; continuous samples avoid exact cost ties almost surely.
Test bijectivity, non-increasing cost and source-swap equivariance. Additionally
train the same D on near-matched **real versus real** batches for 2k updates,
three seeds on 5×5, then evaluate 51,200 fresh pairs per seed. This is a diagnostic
against accidental source-role signals, not a guarantee against every classifier.

D sees near pairs but G sees random pairs. A bad result may therefore reflect
this difference in training contexts as well as whether local comparisons help.
No mode labels or matching outputs are fed into D or G.

## Underrepresented-mode references

The sampler uses the known Gaussian centers and acceptance radius to identify
mode deficits; networks remain unconditional. Extending this to images would
require class labels or a defined feature/clustering proxy.

Let q_k be an exponential moving average of G's accepted mass in mode k,
including invalid points in the denominator. Initialize q_k=1/K. After each
training step, update q from its existing D fake batch with decay 0.99
(approximately a 100-update time scale), requiring no extra generator samples.
Use only the previous estimate when choosing a step's real samples.

Let d_k=max(1/K-q_k,0). D real-mode probabilities are:

    w_k = 0.5/K + 0.5*d_k/sum_j(d_j)

Use uniform probabilities if all deficits vanish. Equal accepted mass across
modes also yields uniform weights, so the ideal symmetric distribution is not
reweighted in the population limit; this does not guarantee stable dynamics. Draw 256 mode indices with
replacement from these weights, then fresh Gaussian noise with the original
sigma. Every mode retains probability at least 0.5/K. No preference parameters
are tuned after viewing results. Log weights and EMA estimates at each evaluation.

Both paired and vanilla use this rule, each driven by its own generator. They
share the intervention algorithm, not an identical sequence of weights. This
changes the real distribution D is trained against, so it is not a pure pairing
ablation or the original fixed-target GAN objective. In paired-deficit, G still
sees uniform real references, which also differs from D’s weighted real context. All evaluations continue
to use the original uniform mixture. Near and deficit arms deliberately test
different mechanisms; do not attribute their difference solely to pair distance.

## Evidence retained

Mode TV, validity, both coverage thresholds, fine-grid density TV, five-seed
learning curves and samples. For near, log mean RMS pair distance before/after
matching. For deficit, retain real sampling weights and generator mass estimates.
Verify endpoint metrics from saved samples, exact checkpoint replay, original
source/configuration provenance, unchanged G random streams and optimizer budgets.
Use `grid_reference_audit`, then `grid_reference_run`, then `grid_reference_report`.
