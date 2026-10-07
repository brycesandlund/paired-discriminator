# Dual-slot D-only deficit sampling (#23)

Prespecified before outcomes. New dual_slot runs use the same deficit rule as
paired D-only (#20), compared with frozen original dual_slot and paired D-only.
Other historical grid arms, including G-only paired (#22), remain in the report.

- Grids 3×3 and 5×5: 50k updates; 7×7: 150k. Seeds 0–4.
- Original spacing 1.5, sigma .1, architectures, Adam, losses, and batch sizes.
- D draws 256 real points independently with replacement from the weighted
  mixture and 256 generated points. Its 256 pairs comprise 64 each RR/FF/FR/RF.
  Both real members of every RR pair come from independent weighted draws.
- The deficit estimate uses all 256 existing D fake points, once each, including
  both fake slots of FF pairs. No new generated samples are added.
- Accepted mass means distance to nearest center <=3 sigma; invalid samples
  stay in the denominator. EMA starts at 1/K and has decay .99.
- At each step, weights use the previous EMA: d=max(1/K-q,0),
  w=.5/K+.5*d/sum(d), uniform fallback. Update EMA after both optimizers.
- G is unchanged: 256 fresh generated points, 64 FF + 64 FR + 64 RF pairs,
  fake-slot BCE only. Draw 256 uniform real references but use only 128, exactly
  as in the frozen baseline. No G reference reweighting.
- One D and one G optimizer step per round. Preserve real-G, noise-D/noise-G,
  slots and evaluation RNG streams; only real-D sampling consumption changes.
- Keep the original uniform target for every metric. Log coarse metrics every
  1k; retain checkpoints/samples at 10k/50k and 100k/150k for 7×7.
- Report five-seed mean/SD, per-seed mode TV, fine-density TV, valid mass,
  >=1% coverage, and existing relative coverage. Retain all failed seeds.

Tests check the unchanged G real batches and RNG streams, complete weighted
real-point allocation to RR/FR/RF, pair composition and optimizer counts.
Post-run verification replays all 40 retained endpoints, checks optimizer steps
and unchanged RNG states against original dual_slot, and recomputes historical
metrics. Source snapshots/configuration accompany each grid result directory.

```sh
uv run pytest tests/test_grid_dual_deficit.py -q
uv run python -m paired_discriminator.grid_dual_deficit_run
```

Fifteen local runs, eight processes with one torch thread each. The launcher
builds `results/grid-dual-deficit-comparison-v1/REPORT.md` after completion.
Training uses an isolated module; historical sources and results are unchanged.
