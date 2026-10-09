# CIFAR optimizer and EMA tuning (#37)

Exploratory tuning, unconditional and uniform. Warm branches start from GAN50k or flow80k with Adam and explicit sampling streams reset identically, including controls. Raw and EMA weights evaluated with the same10k noise samples. Primary screening criterion: training-reference FID. Test-reference scores are descriptive, not an untouched final test. Classifier TV and recall are reported alongside fidelity.

## Every evaluated candidate

| Arm / step | Weights | Train FID ↓ | Test FID ↓ | KID ↓ | Precision | Recall | VGG TV ↓ | ResNet TV ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| confirm_flow_lr5e5_seed2/eval_030000_model | model | 42.11 | 41.77 | 0.03129 | 0.618 | 0.365 | 0.150 | 0.170 |
| flow_control_seed0/eval_030000_model | model | 42.39 | 42.11 | 0.03505 | 0.605 | 0.381 | 0.145 | 0.138 |
| confirm_flow_control_seed1/eval_030000_ema | ema | 42.90 | 42.57 | 0.03381 | 0.626 | 0.367 | 0.124 | 0.145 |
| flow_control_seed0/eval_020000_model | model | 42.98 | 42.65 | 0.03427 | 0.598 | 0.363 | 0.170 | 0.160 |
| confirm_flow_lr5e5_seed1/eval_030000_ema | ema | 43.11 | 42.78 | 0.03355 | 0.628 | 0.371 | 0.123 | 0.142 |
| confirm_flow_lr5e5_seed2/eval_030000_ema | ema | 43.17 | 42.81 | 0.03348 | 0.628 | 0.378 | 0.128 | 0.150 |
| confirm_flow_control_seed2/eval_030000_ema | ema | 43.21 | 42.86 | 0.03392 | 0.624 | 0.368 | 0.127 | 0.149 |
| flow_lr5e5_seed0/eval_030000_ema | ema | 43.67 | 43.33 | 0.03438 | 0.618 | 0.351 | 0.119 | 0.140 |
| flow_control_seed0/eval_030000_ema | ema | 43.68 | 43.36 | 0.03475 | 0.611 | 0.340 | 0.121 | 0.140 |
| flow_lr5e5_seed0/eval_020000_ema | ema | 44.01 | 43.67 | 0.03480 | 0.620 | 0.347 | 0.120 | 0.142 |
| flow_lr5e5_seed0/eval_030000_model | model | 44.08 | 43.74 | 0.03579 | 0.609 | 0.360 | 0.118 | 0.122 |
| confirm_flow_control_seed2/eval_030000_model | model | 44.13 | 43.74 | 0.03361 | 0.619 | 0.348 | 0.179 | 0.181 |
| saved_flow_ema_100000 | ema | 44.20 | 43.87 | 0.03538 | 0.615 | 0.349 | 0.118 | 0.144 |
| flow_lr5e5_seed0/eval_010000_ema | ema | 44.23 | 43.90 | 0.03491 | 0.622 | 0.344 | 0.123 | 0.148 |
| flow_control_seed0/eval_020000_ema | ema | 44.29 | 43.95 | 0.03560 | 0.617 | 0.342 | 0.121 | 0.140 |
| flow_lr1e4_seed0/eval_010000_ema | ema | 44.41 | 44.09 | 0.03524 | 0.619 | 0.344 | 0.120 | 0.149 |
| flow_lr5e5_seed0/eval_020000_model | model | 44.52 | 44.12 | 0.03615 | 0.608 | 0.357 | 0.154 | 0.153 |
| flow_decay_seed0/eval_010000_ema | ema | 44.67 | 44.35 | 0.03569 | 0.618 | 0.344 | 0.124 | 0.148 |
| flow_control_seed0/eval_010000_ema | ema | 44.89 | 44.57 | 0.03600 | 0.622 | 0.341 | 0.126 | 0.152 |
| flow_decay_seed0/eval_010000_model | model | 45.22 | 44.94 | 0.03610 | 0.608 | 0.350 | 0.155 | 0.159 |
| base_flow_control_seed1/eval_080000_ema | ema | 45.33 | 44.97 | 0.03580 | 0.625 | 0.347 | 0.129 | 0.153 |
| base_flow_control_seed2/eval_080000_ema | ema | 45.40 | 45.07 | 0.03605 | 0.633 | 0.361 | 0.131 | 0.158 |
| flow_lr1e4_seed0/eval_010000_model | model | 45.44 | 45.15 | 0.03623 | 0.620 | 0.329 | 0.148 | 0.161 |
| flow_lr5e5_seed0/eval_010000_model | model | 45.52 | 45.16 | 0.03676 | 0.608 | 0.349 | 0.119 | 0.143 |
| saved_flow_ema_80000 | ema | 45.83 | 45.50 | 0.03682 | 0.627 | 0.334 | 0.125 | 0.155 |
| flow_control_seed0/eval_010000_model | model | 46.13 | 45.81 | 0.03730 | 0.604 | 0.348 | 0.185 | 0.172 |
| base_flow_control_seed2/eval_080000_model | model | 46.65 | 46.41 | 0.03949 | 0.603 | 0.349 | 0.126 | 0.140 |
| confirm_flow_lr5e5_seed1/eval_030000_model | model | 47.00 | 46.65 | 0.03771 | 0.630 | 0.357 | 0.118 | 0.126 |
| saved_flow_ema_50000 | ema | 49.35 | 49.03 | 0.04013 | 0.631 | 0.318 | 0.140 | 0.171 |
| confirm_flow_control_seed1/eval_030000_model | model | 49.84 | 49.50 | 0.04306 | 0.620 | 0.361 | 0.153 | 0.173 |
| base_flow_control_seed1/eval_080000_model | model | 50.14 | 49.64 | 0.04101 | 0.627 | 0.326 | 0.180 | 0.182 |
| paired_control_seed0/eval_020000_ema | ema | 48.39 | 48.23 | 0.03310 | 0.566 | 0.352 | 0.162 | 0.213 |
| paired_control_seed0/eval_030000_ema | ema | 48.62 | 48.42 | 0.03386 | 0.573 | 0.348 | 0.161 | 0.209 |
| confirm_paired_control_seed2/eval_020000_ema | ema | 48.97 | 48.75 | 0.03448 | 0.579 | 0.337 | 0.161 | 0.204 |
| confirm_paired_control_seed1/eval_020000_ema | ema | 49.59 | 49.32 | 0.03417 | 0.558 | 0.348 | 0.158 | 0.201 |
| paired_control_seed0/eval_010000_ema | ema | 49.68 | 49.48 | 0.03368 | 0.563 | 0.342 | 0.142 | 0.201 |
| paired_control_seed0/eval_020000_model | model | 49.75 | 49.62 | 0.03447 | 0.570 | 0.336 | 0.164 | 0.231 |
| paired_slow_d_seed0/eval_010000_ema | ema | 49.91 | 49.72 | 0.03395 | 0.550 | 0.356 | 0.135 | 0.192 |
| paired_decay_seed0/eval_010000_ema | ema | 50.04 | 49.85 | 0.03411 | 0.553 | 0.337 | 0.135 | 0.205 |
| paired_control_seed0/eval_030000_model | model | 50.04 | 49.84 | 0.03521 | 0.580 | 0.326 | 0.175 | 0.216 |
| paired_low_both_seed0/eval_010000_model | model | 50.16 | 50.02 | 0.03465 | 0.558 | 0.320 | 0.153 | 0.222 |
| paired_low_both_seed0/eval_010000_ema | ema | 50.27 | 50.06 | 0.03405 | 0.565 | 0.341 | 0.141 | 0.204 |
| paired_control_seed0/eval_010000_model | model | 50.37 | 50.27 | 0.03513 | 0.545 | 0.330 | 0.152 | 0.221 |
| paired_decay_seed0/eval_010000_model | model | 50.38 | 50.19 | 0.03452 | 0.545 | 0.332 | 0.144 | 0.214 |
| confirm_paired_control_seed2/eval_020000_model | model | 50.42 | 50.12 | 0.03542 | 0.585 | 0.325 | 0.144 | 0.195 |
| paired_slow_g_seed0/eval_010000_ema | ema | 50.70 | 50.52 | 0.03428 | 0.562 | 0.328 | 0.154 | 0.214 |
| paired_slow_d_seed0/eval_010000_model | model | 50.79 | 50.61 | 0.03493 | 0.555 | 0.327 | 0.151 | 0.214 |
| confirm_paired_control_seed1/eval_020000_model | model | 51.01 | 50.77 | 0.03483 | 0.560 | 0.331 | 0.160 | 0.215 |
| base_paired_control_seed1/eval_050000_ema | ema | 51.01 | 50.76 | 0.03434 | 0.531 | 0.342 | 0.145 | 0.198 |
| paired_slow_g_seed0/eval_010000_model | model | 51.22 | 51.09 | 0.03478 | 0.552 | 0.326 | 0.158 | 0.233 |
| base_paired_control_seed1/eval_050000_model | model | 53.07 | 52.86 | 0.03590 | 0.542 | 0.298 | 0.167 | 0.207 |
| base_paired_control_seed2/eval_050000_ema | ema | 53.74 | 53.43 | 0.03758 | 0.567 | 0.299 | 0.155 | 0.224 |
| base_paired_control_seed2/eval_050000_model | model | 55.30 | 55.05 | 0.03908 | 0.572 | 0.307 | 0.163 | 0.225 |
| paired_r1_seed0/eval_010000_model | model | 57.67 | 57.58 | 0.04019 | 0.482 | 0.248 | 0.137 | 0.214 |
| paired_r1_seed0/eval_010000_ema | ema | 59.80 | 59.62 | 0.04150 | 0.465 | 0.261 | 0.137 | 0.193 |
| base_vanilla_control_seed1/eval_050000_ema | ema | 48.96 | 48.73 | 0.03441 | 0.555 | 0.350 | 0.194 | 0.237 |
| vanilla_control_seed0/eval_010000_ema | ema | 49.68 | 49.58 | 0.03681 | 0.586 | 0.364 | 0.246 | 0.270 |
| vanilla_slow_d_seed0/eval_010000_ema | ema | 49.69 | 49.61 | 0.03632 | 0.584 | 0.361 | 0.239 | 0.270 |
| vanilla_low_both_seed0/eval_010000_ema | ema | 49.76 | 49.65 | 0.03678 | 0.592 | 0.373 | 0.236 | 0.265 |
| base_vanilla_control_seed2/eval_050000_ema | ema | 50.24 | 50.05 | 0.03760 | 0.580 | 0.338 | 0.191 | 0.230 |
| vanilla_decay_seed0/eval_010000_ema | ema | 50.29 | 50.18 | 0.03700 | 0.585 | 0.363 | 0.243 | 0.270 |
| vanilla_slow_g_seed0/eval_010000_ema | ema | 50.32 | 50.19 | 0.03745 | 0.590 | 0.356 | 0.241 | 0.269 |
| vanilla_low_both_seed0/eval_010000_model | model | 50.38 | 50.28 | 0.03734 | 0.589 | 0.365 | 0.231 | 0.262 |
| base_vanilla_control_seed1/eval_050000_model | model | 50.42 | 50.21 | 0.03501 | 0.567 | 0.350 | 0.210 | 0.267 |
| confirm_vanilla_control_seed1/eval_010000_ema | ema | 50.73 | 50.60 | 0.03547 | 0.579 | 0.360 | 0.239 | 0.270 |
| vanilla_slow_g_seed0/eval_010000_model | model | 51.20 | 51.07 | 0.03804 | 0.591 | 0.362 | 0.246 | 0.280 |
| vanilla_slow_d_seed0/eval_010000_model | model | 51.34 | 51.25 | 0.03796 | 0.581 | 0.345 | 0.241 | 0.272 |
| vanilla_control_seed0/eval_010000_model | model | 51.60 | 51.51 | 0.03819 | 0.588 | 0.362 | 0.258 | 0.290 |
| vanilla_control_seed0/eval_020000_ema | ema | 51.91 | 51.87 | 0.03831 | 0.600 | 0.359 | 0.279 | 0.317 |
| base_vanilla_control_seed2/eval_050000_model | model | 51.96 | 51.80 | 0.03875 | 0.589 | 0.341 | 0.202 | 0.243 |
| vanilla_decay_seed0/eval_010000_model | model | 52.00 | 51.89 | 0.03858 | 0.592 | 0.354 | 0.244 | 0.277 |
| confirm_vanilla_control_seed2/eval_010000_ema | ema | 52.31 | 52.17 | 0.03932 | 0.599 | 0.343 | 0.237 | 0.262 |
| confirm_vanilla_control_seed1/eval_010000_model | model | 52.90 | 52.86 | 0.03739 | 0.576 | 0.337 | 0.232 | 0.275 |
| vanilla_control_seed0/eval_020000_model | model | 53.91 | 53.88 | 0.03951 | 0.605 | 0.349 | 0.287 | 0.337 |
| vanilla_control_seed0/eval_030000_ema | ema | 54.67 | 54.61 | 0.04085 | 0.611 | 0.349 | 0.319 | 0.355 |
| confirm_vanilla_control_seed2/eval_010000_model | model | 54.80 | 54.69 | 0.04182 | 0.608 | 0.308 | 0.250 | 0.270 |
| vanilla_control_seed0/eval_030000_model | model | 57.85 | 57.88 | 0.04452 | 0.609 | 0.337 | 0.326 | 0.363 |
| vanilla_r1_seed0/eval_010000_ema | ema | 75.61 | 75.71 | 0.06056 | 0.391 | 0.203 | 0.204 | 0.290 |
| vanilla_r1_seed0/eval_010000_model | model | 77.99 | 78.39 | 0.06355 | 0.282 | 0.135 | 0.287 | 0.294 |
