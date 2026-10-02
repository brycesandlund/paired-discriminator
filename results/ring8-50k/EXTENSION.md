# Extension: 10,000 → 50,000 updates

Same five seeds, models, optimizer settings, random streams, and evaluation samples. The deterministic training run was replayed from initialization and extended; all 110 original evaluation rows match exactly (excluding timing).

## Endpoint comparison

Mean ± sample SD across five seeds. Lower mode TV is better; it penalizes both incorrect mode masses and invalid samples.

| Method | Updates | Modes covered | Valid samples | Mode TV |
|---|---:|---:|---:|---:|
| vanilla | 10,000 | 7.60 ± 0.89 | 84.9% ± 1.2% | 0.226 ± 0.072 |
| vanilla | 50,000 | 6.80 ± 0.84 | 92.7% ± 1.6% | 0.183 ± 0.074 |
| paired | 10,000 | 8.00 ± 0.00 | 83.1% ± 0.3% | 0.169 ± 0.003 |
| paired | 50,000 | 8.00 ± 0.00 | 93.1% ± 2.0% | 0.069 ± 0.020 |

## Seed-level endpoints

| Method | Seed | Modes at 10k → 50k | Valid at 10k → 50k | TV at 10k → 50k |
|---|---:|---:|---:|---:|
| vanilla | 0 | 8 → 8 | 86.0% → 93.7% | 0.176 → 0.100 |
| vanilla | 1 | 8 → 6 | 83.9% → 89.9% | 0.262 → 0.250 |
| vanilla | 2 | 6 → 6 | 83.6% → 93.7% | 0.335 → 0.272 |
| vanilla | 3 | 8 → 7 | 84.9% → 93.7% | 0.193 → 0.151 |
| vanilla | 4 | 8 → 7 | 86.2% → 92.6% | 0.164 → 0.142 |
| paired | 0 | 8 → 8 | 83.0% → 93.6% | 0.170 → 0.064 |
| paired | 1 | 8 → 8 | 83.3% → 93.8% | 0.167 → 0.062 |
| paired | 2 | 8 → 8 | 83.2% → 94.5% | 0.168 → 0.055 |
| paired | 3 | 8 → 8 | 83.1% → 93.9% | 0.169 → 0.061 |
| paired | 4 | 8 → 8 | 82.7% → 89.5% | 0.173 → 0.105 |

## Late training: 40,000–50,000 updates

These are averages over the recorded late-window evaluations within each seed, then averaged across seeds. Checkpoints are correlated, not additional independent replicates. Full-coverage checks refer only to recorded evaluations, not every training step.

| Method | Mean coverage | Mean validity | Mean TV | Full-coverage checks |
|---|---:|---:|---:|---:|
| vanilla | 6.80 | 92.7% | 0.187 | 20.0% |
| paired | 8.00 | 93.1% | 0.070 | 100.0% |

## Plots

![Training curves](training_curves.png)

![Final generated samples](final_samples.png)

![Accepted mode mass](mode_mass.png)

## Limits

This extends one untuned setup. It does not prove convergence, general superiority, or that the proposed reference-dependent gradient mechanism explains any difference. Both methods see 256 real and 256 generated samples in each discriminator update; unary makes two decisions per pair, while paired makes one joint decision. Full metric definitions and parameter counts are in REPORT.md.
