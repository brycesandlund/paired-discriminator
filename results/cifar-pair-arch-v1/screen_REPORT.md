# Paired CIFAR architecture comparisons (#38)

Collected 36 evaluations. 6/6 seed-0 primary endpoints available.

Primary comparison: fresh training for 50k updates, EMA weights, train-reference FID for architecture selection. Raw weights and earlier checkpoints are descriptive. Same generator, BCE, Adam settings, batches, real draws and evaluation noise. Training compute differs and is measured. Held-out reference scores are exploratory, not an untouched final test.

## Seed-0 primary comparison

| Architecture | Train FID ↓ | Test FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cross-attention | 48.17080 | 48.08272 | 0.03312 | 0.56000 | 0.39610 | 0.22870 | 0.24660 |
| Shared independent scores | 49.32384 | 49.18964 | 0.03445 | 0.56730 | 0.33810 | 0.17980 | 0.21540 |
| Wider concat | 50.59734 | 50.33014 | 0.03440 | 0.56860 | 0.32350 | 0.16390 | 0.22170 |
| Original concat | 51.78318 | 51.55566 | 0.03484 | 0.54640 | 0.30170 | 0.14800 | 0.21160 |
| Global reference context | 52.67851 | 52.58117 | 0.03809 | 0.60410 | 0.35390 | 0.26430 | 0.26490 |
| Self-attention control | 52.86391 | 52.80322 | 0.03787 | 0.59090 | 0.36790 | 0.27020 | 0.28950 |

## Training trajectories

![Training trajectories](trajectories.png)

## Reference interaction diagnostics

Hold fake pixels fixed and change the real reference. Additive unary scoring has the same raw-logit gradient direction even though BCE can rescale it. A lower direction cosine or a nonzero four-pair interaction residual shows reference dependence, not necessarily useful comparison or diversity.

| Architecture | Step | Direction cosine | Interaction RMS | Logit RMS | Swap error |
|---|---:|---:|---:|---:|---:|
| Original concat | 10,000 | 0.40123 | 0.48429 | 2.23541 | 5.9238 |
| Original concat | 20,000 | 0.39838 | 0.33612 | 2.08036 | 11.095 |
| Original concat | 50,000 | 0.32487 | 0.55404 | 3.33832 | 22.673 |
| Wider concat | 10,000 | 0.41848 | 0.63843 | 2.14571 | 7.993 |
| Wider concat | 20,000 | 0.37982 | 0.42610 | 1.54757 | 6.559 |
| Wider concat | 50,000 | 0.17675 | 0.45181 | 3.09934 | 17.295 |
| Shared independent scores | 10,000 | 1.00000 | 0.00000 | 1.91313 | 0 |
| Shared independent scores | 20,000 | 1.00000 | 0.00000 | 1.69598 | 0 |
| Shared independent scores | 50,000 | 1.00000 | 0.00000 | 4.89137 | 0 |
| Global reference context | 10,000 | 0.94822 | 0.71198 | 5.43955 | 0 |
| Global reference context | 20,000 | 0.90056 | 0.68742 | 5.37261 | 0 |
| Global reference context | 50,000 | 0.90303 | 0.92055 | 8.33270 | 0 |
| Self-attention control | 10,000 | 1.00000 | 0.00000 | 4.11776 | 0 |
| Self-attention control | 20,000 | 1.00000 | 0.00000 | 5.22706 | 0 |
| Self-attention control | 50,000 | 1.00000 | 0.00000 | 7.03934 | 0 |
| Cross-attention | 10,000 | 0.93036 | 0.94208 | 3.24704 | 0 |
| Cross-attention | 20,000 | 0.92695 | 0.73699 | 3.70556 | 0 |
| Cross-attention | 50,000 | 0.90467 | 1.06557 | 4.68398 | 0 |

## Cost and provenance

Recorded training 1.66 GPU-hours + evaluation 0.34 = 2.00 GPU-hours. Excludes startup, checkpoint/volume I/O and short verification jobs.

| Architecture | D parameters | Completed steps | Training minutes |
|---|---:|---:|---:|
| concat_seed0 | 666,049 | 50,000 | 11.21 |
| concat_wide_seed0 | 707,983 | 50,000 | 14.12 |
| cross_attention_seed0 | 712,770 | 50,000 | 21.42 |
| global_context_seed0 | 712,770 | 50,000 | 17.84 |
| self_attention_seed0 | 712,770 | 50,000 | 20.39 |
| shared_difference_seed0 | 662,977 | 50,000 | 14.87 |

Shared CNN weights are initialized identically across the shared-encoder arms. Self-attention and cross-attention have identical initial parameters; only their source of keys/values differs. Global conditioning has the same total parameter count. All shared arms have an antisymmetric output. Independent shared scoring and self-attention remain additive unary controls.

Attention runs over 8×8 tokens with four heads. Residual scale starts at 0.1 and is learned. The wider concatenation control has about 0.7% fewer parameters than the interaction arms. All methods draw 128 real and fake images for D and another 128 real references and fake images for G each update. At 50k this is 12.8M real draws per arm. Generator architecture and one-pass inference cost are unchanged.

## Fixed samples at the primary endpoint

![Primary samples](samples_050000_ema.png)

All 64 fixed samples; no filtering. A comparison across architectures uses the same latent noise, but this does not align generated semantic content.
