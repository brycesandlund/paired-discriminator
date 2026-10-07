# 10×10 grid: uniform versus D-only deficit (#24)

Prespecified before outcomes. Four arms: vanilla, paired, vanilla_deficit,
paired_deficit. Seeds 0–4, 150k D/G updates. Retain endpoints at 10k, 50k,
100k, 150k and coarse metrics every 1k. Twenty new trajectories; no extensions
or architecture/hyperparameter tuning within this experiment.

## Geometry and workload

100 isotropic Gaussians, sigma .1, spacing 1.5. Centers span [-6.75,6.75]
on both axes: the grid gets larger, not denser. Networks, latent dimension,
optimizer, batch 256, one D/G update per step, and fixed 10k evaluation draws
match earlier grids. D consumes 256 real and 256 generated points. G consumes
256 fresh generated points; paired references stay independently uniform.

D-only adaptive arms reuse existing D fake samples to estimate accepted mode
mass (nearest center within 3 sigma; invalid samples remain in denominator).
EMA starts at 1/100 and decays .99. D real sampling uses the previous EMA:
w_k=.5/100+.5*max(.01-q_k,0)/sum positive deficits, uniform fallback. Update EMA
after both optimizers. Each arm estimates its own deficits. Only adaptive D real
sampling changes; G/noise/slot/evaluation RNGs are preserved versus each control.

## Metrics and coverage interpretation

Mode TV includes invalid mass and targets the original uniform mixture.
Fine-density TV keeps 0.05-wide cells but expands the region to [-7.75,7.75]²,
plus an outside category. The old ±5.5 bounds would incorrectly pool outer modes.
Calibrate both TV metrics and coverage against 200 independent true-mixture draws
of 10k samples. Finite-sample density TV grows with complexity, so raw values
across grid sizes are not directly comparable as model error.

Coverage thresholds, all divided by the total generated sample count:

- Legacy 1% mass: retained, but equals 100% of each mode's target on this grid.
  Even perfect finite-sample draws should not be expected to satisfy it everywhere.
- >=50% target mass: 0.005, at least 50 accepted samples per mode out of 10k.
  Added as an interpretable stricter relative criterion, before inspecting results.
- Existing >=8% target mass: 0.0008, at least eight samples; a lenient presence test.

Report all seeds, means/SD, full-coverage seed counts, accepted-mass heatmaps,
and fixed seed-0 scatter plots. Neither coarse coverage nor averaged heatmaps
establish correct within-mode density or coverage in every seed.

## Reproducibility

Reuse the tested `grid_reference.run` loop without changing it. Tests cover new
geometry, density bounds and threshold semantics. Snapshot source/configuration;
replay all 80 saved endpoints exactly, verify Adam step counts, and compare the
40 adaptive-arm G/noise/slot/evaluation RNG states against their uniform controls.

```sh
uv run pytest -q
uv run python -m paired_discriminator.grid10_run
```

Eight local workers, one torch thread each. Results in `results/grid10-150k-v1`;
automatic verified report in `results/grid10-comparison-v1/REPORT.md`.
