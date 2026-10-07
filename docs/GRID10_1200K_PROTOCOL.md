# 10×10 continuation to 1.2M (#27)

Continue all four methods × seeds 0–4 from 600k to 1,200,000 updates.
Restore networks, Adam optimizers, RNG streams and deficit EMA with the tested
exact-continuation trainer. All training settings remain unchanged.

Evaluate 10,000 fixed samples every 50k updates, retaining full checkpoints.
Replay the starting 600k boundary. Reporting verifies 260 endpoint replays,
130 adaptive/control RNG comparisons, source hashes and optimizer counts.
Prior experiments remain preserved. Eight local workers, one torch thread each.

```sh
uv run python -m paired_discriminator.grid10_1200k_run
```

Results: `results/grid10-1200k-v1`; automatic report:
`results/grid10-1200k-comparison-v1/REPORT.md`.
A thread completion check will report completion or failure.
