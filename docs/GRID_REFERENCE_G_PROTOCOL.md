# Grid G-only deficit reference experiment (#22)

Prespecified before reviewing outcomes. Tests the user's intended intervention:
keep D's real sampling uniform, bias only the real references used in paired G
updates toward modes underrepresented by G.

## Fixed protocol

- One new paired arm, seeds 0–4, 3×3/5×5 grids to 50k and 7×7 to 150k.
- Original spacing 1.5, Gaussian sigma .1, networks, losses, Adam, D/G batches
  (256), optimizer counts, and evaluation draws remain unchanged.
- D receives 256 uniform real points and 256 generated points each update.
- G receives 256 fresh generated points and 256 independently sampled real
  references from the deficit-weighted mixture. No per-fake matching or
  reconstruction objective; paired BCE targets and randomized slots unchanged.
- EMA accepted generated mass q starts at 1/K. Acceptance is nearest center
  within 3 sigma; invalid generated points remain in the denominator.
- Reuse the existing D fake batch to observe mass. No additional G samples.
- Reference weights use the previous EMA: d=max(1/K-q,0),
  w=0.5/K+0.5*d/sum(d), uniform fallback if no deficit. After both optimizers,
  update q=0.99*q+0.01*observed mass. This timing matches the D-only experiment.
- Only real-G RNG consumption changes. D real draws, noise-D/noise-G streams,
  slot RNG and evaluation draws match original paired runs.

## Comparisons and interpretation

Reuse frozen original vanilla, RSGAN, paired, PacGAN2, dual-slot, and prior D-only
reference experiments. Focus on original paired vs D-only paired deficit vs
G-only paired deficit. Vanilla lacks a real-reference G input, so no new vanilla
arm is trained. Baselines are reused at matching seed, grid and step.

Evaluate mode TV, fine-grid density TV, valid mass, >=1% coverage, and the
existing relative coverage threshold every applicable retained endpoint: 10k,
50k, and for 7×7 also 100k/150k. Mode metrics are logged every 1k. All metrics
retain the original uniform mixture as target; all seeds/failures are retained.
Five-seed means/SDs and per-seed endpoint CSVs avoid selecting a favorable seed.
Heatmaps show mean accepted mass relative to target; they do not by themselves
establish that each seed covers every mode.

This remains an oracle mode-level intervention using known Gaussian geometry,
not the class-free within-mode sampler discussed for natural images. D fake
inputs can change as G learns differently; 'D unchanged' means its objective,
real sampling, and workload are unchanged, not identical learned weights.

## Implementation and verification

`grid_reference_g.py` preserves the prior loop as an isolated source module and
changes the sampling phase. Prior source/results remain untouched. Tests check
exact D real batches, unchanged D/noise/slot/evaluation RNG states, one weighted
G reference batch per step, and optimizer counts. The report replays every new
retained checkpoint, checks unchanged RNG streams against old paired, and
recomputes historical metrics from saved samples. Source snapshots and hashes
are retained alongside configuration and checkpoints.

```sh
uv run pytest tests/test_grid_reference_g.py tests/test_grid_reference.py -q
uv run python -m paired_discriminator.grid_reference_g_run
```

15 local runs, eight worker processes with one torch thread each. The launcher
automatically builds `results/grid-reference-g-comparison-v1/REPORT.md` when
all runs finish. Per-grid results: `results/grid{3,5,7}-reference-g-v1`.
