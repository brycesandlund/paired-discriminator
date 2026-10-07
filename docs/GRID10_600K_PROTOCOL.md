# 10×10 continuation to 600k (#26)

Continue all 20 runs from their completed 300k checkpoints to 600k, restoring
both networks, both Adam optimizers, all explicit RNG streams and deficit EMA.
Use the existing tested exact-continuation loop. No training hyperparameter,
geometry, target, sample budget per update, or evaluation-definition changes.

Retain fixed 10k-sample evaluations and full checkpoints at 350k, 400k, 450k,
500k, 550k and 600k, plus an exact replay of the 300k starting endpoint.
The original 150k and 300k experiment directories are preserved.

## Evaluation cost and cadence

Measured on the local machine, single torch thread, 2026-10-07:

- Median generation of 10k samples + coarse/fine TV + checkpoint serialization
  to memory: 0.00339 seconds (ten measured repetitions after two warmups).
- Previous 150k→300k run median non-training wall time: 0.151 seconds per run,
  about 0.097% of wall time; this includes evaluation and checkpoint disk writes.
- The post-run verification, calibration, and plotting also take time; the
  percentage above describes training-loop overhead, not all reporting work.

50k cadence is therefore inexpensive and gives useful resolution; there is no
need to reduce it to 100k. Raw measurements: `results/grid10-eval-cost.json`.

Verification repeats source/checkpoint hashes, all 20 boundary sample replays,
140 endpoint model/sample replays (including the 20 starting endpoints), Adam
step counts, and 70 adaptive/control G/noise/slot/evaluation RNG comparisons.
All seeds remain; the 600k endpoint is not assumed to be convergence.

```sh
uv run python -m paired_discriminator.grid10_600k_run
```

Twenty local runs, eight workers and one torch thread each. Results:
`results/grid10-600k-v1`; automatic report:
`results/grid10-600k-comparison-v1/REPORT.md`.
