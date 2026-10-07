# 10×10 exact continuation to 300k (#25)

Continue all 20 trajectories from #24: vanilla, paired, vanilla D-deficit,
paired D-deficit; seeds 0–4. Resume the saved 150k checkpoint and train another
150k steps. Evaluate and retain complete checkpoints plus the same fixed 10k
sample arrays at 200k, 250k and 300k. Also replay/store the 150k starting endpoint.

Restore both networks, both Adam optimizers (including moments and step counts),
all six explicit RNG streams, and the deficit EMA. Recreate fixed evaluation
noise from its original seed before restoring the checkpointed eval RNG.
No changes to networks, distributions, batches, learning rate, objectives,
EMA decay, uniform floor or geometry. Only target steps and evaluation cadence
change. G references remain uniform. Checkpoint sources and SHA256 are recorded.

An isolated continuation module leaves earlier source/results intact. Four
small exact-replay tests compare interrupted vs uninterrupted runs, including
optimizer tensors and EMA, and use differing evaluation schedules. The report
also verifies every replayed 150k sample array against the original run, all
80 retained endpoint arrays against their generator states, Adam counts, source
hashes, and the 40 adaptive/control G/noise RNG comparisons.

Metrics retain the 10×10 evaluation definitions: ±7.75 density bounds, .05
cells, invalid-mass-aware mode TV, half-target and 8%-target coverage. No claim
that 300k is sufficient for convergence is assumed in advance. All seeds remain.

```sh
uv run pytest tests/test_grid10_continue.py -q
uv run python -m paired_discriminator.grid10_continue_run
```

Results: `results/grid10-300k-v1`; automatic report:
`results/grid10-300k-comparison-v1/REPORT.md`. Original 150k results are preserved.
