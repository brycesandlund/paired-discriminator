# Tuned GANs with uniform real sampling (#34)

User-requested follow-up to #33: freeze the successful feature architecture and
optimizer schedule, turn off deficit sampling, and compare vanilla and paired
across seeds0–4 at 50k. No additional hyperparameter search in this experiment.
Reuse the exact paired-uniform seed0 screen; train paired seeds1–4 and vanilla
seeds0–4. This gives five trajectories per method with nine new training jobs.

Both models use the 2D normal latent Fourier residual G, progressive Fourier D,
two width128 LeakyReLU hidden layers, Adam (.5,.999), and lr .001 for 25k then
cosine decay to .0001 by 50k. Batch256 for each real/fake draw. D draws real
points uniformly from the target mixture (alpha=0); paired G references are
also uniform. No reconstruction or other auxiliary objective. Mode identities
are used to evaluate samples, but do not change training sampling or losses.
The trainer still computes an unused deficit estimate to preserve its existing
code path; alpha=0 bypasses weighted real sampling.

Raw weights are primary. Retain generator EMA(.999) as a fixed secondary output
for every seed. Checkpoints at 25k and 50k; final comparison uses exactly the
same 10k latent samples per seed as #33 (seed932000+training_seed). Replay the
saved seed70000+training_seed arrays as a check, then compare all four arms:
vanilla uniform, paired uniform, vanilla linear deficit, paired linear deficit.
Reuse all deficit checkpoints from #33; no new deficit training. Report means,
sample standard deviations, per-seed values, full half-target coverage counts,
and seed0 samples without selecting a seed or checkpoint for appearance.

All four arms share G architecture, training updates and nominal batch sizes.
Paired has a larger joint D and uses extra G real references. Hence equal
updates are not equal compute. The independent 100k-real-point scale pilot is
shared. Source snapshots, checkpoint replays, optimizer counts, and matched RNG
states should be verified. Preserve earlier reports; record this as #34 in the
experiment ledger. No commits.
