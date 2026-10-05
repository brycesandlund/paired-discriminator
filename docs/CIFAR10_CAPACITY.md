# CIFAR-10: discriminator batch and capacity probes

Two new paired-only trajectories, seed 0, 100,000 updates with generator and
discriminator snapshots and integrity evaluations every 10,000 updates. Reuse
all three completed `cifar10-integrity-v1` trajectories as comparison baselines;
no baseline retraining or checkpoint selection.

| Arm | D batch (pairs) | D width / hidden channels | G update batch | G width |
|---|---:|---|---:|---:|
| Existing paired | 128 | 64 / 64,128,256 | 128 | 64 |
| `d_batch256` | 256 | 64 / 64,128,256 | 128 | 64 |
| `d_width128` | 128 | 128 / 128,256,512 | 128 | 64 |

Both arms preserve G architecture, initialization, optimizer, learning rate,
update batch, and one G optimizer step per D optimizer step. G has 1,191,683
parameters. D has 664,769 parameters at width64 and 2,640,257 at width128. D batch256 starts with exactly the same D weights as existing paired.
The wider D has a different initialization because its tensors have different
shapes; projection embedding dimension grows from 256 to 512. Projection remains
an unnormalized dot product with global sum pooling, so widening changes score
and gradient scale as well as capacity.

Only D's batch is doubled in the batch arm: 256 real plus 256 generated samples
per D update. G's own update uses 128 fresh generated samples and 128 fresh real
references in both arms. Extra D fakes are generated in two no-gradient G forwards
of 128, preserving G forward batch size. As in prior experiments, G remains in
training mode during those forwards: the larger-D-batch arm therefore updates
G BatchNorm running statistics three times per iteration instead of twice.
This is a documented side effect, not an additional G optimizer update. No
BatchNorm behavior or evaluation calibration is changed to hide it.

The G noise, requested-label and reference-image RNG streams remain separate
from D's streams. The historical shared slot-permutation stream is retained;
changing D batch consumes more of it, so G slot assignments differ from baseline.
Each update still balances real-in-A / real-in-B exactly. Larger D batches also
change the sample stream grouping; examples are drawn with replacement.

D batch256 approximately matches vanilla's number of hidden feature maps during
its D update (256). It does not equate total compute: the first convolution sees
six channels, extra fakes require extra G forward work, and G updates still use
128 pairs. Wider D uses roughly four times the D parameters/compute and is an
independent capacity probe, not a compute-matched comparison. Counting convolution multiply-accumulates for the D forward only:
vanilla batch256 is 4.496 billion, paired batch128 is 2.349 billion, paired
batch256 is 4.698 billion, and paired width128 batch128 is 8.992 billion.
These exclude activation/pooling/projection work, backward passes, and G.
Both tests are
single-seed, untuned interventions; unchanged Adam 2e-4, betas (0.5,0.999), BCE,
class matching, no augmentation, no spectral normalization, no gradient penalty.

## Evaluation and integrity

Use exactly the [integrity protocol](CIFAR10_INTEGRITY.md): 10,000 fixed generated
samples; FID, KID, precision and recall against fixed training references and the
official held-out test set; D train/held-out comparisons; exhaustive Inception and
pixel nearest neighbors; exact training-copy hashes. The test set remains unused
for gradient updates but is now a diagnostic set, not a pristine final test for
future model selection. All methods use the same generated noise/labels and real
reference sets. The evaluator differs from the previous one only in importing
the model factory that understands independent D width.

Validation covers actual forward batch sizes and gradient-enabled G batch size,
identical G initialization, identical D initialization for batch256, exact CPU
and CUDA checkpoint resume, and exact baseline training replay with unchanged
settings. Historical training and evaluation source files remain untouched.

## Run

```sh
uv run --extra cifar pytest -q
uv run --extra cifar modal run --detach modal_capacity.py::main --run-id cifar10-capacity-v1 --steps 100000
uv run --extra cifar modal run --detach modal_capacity.py::score_range --run-id cifar10-capacity-v1
uv run --extra cifar modal volume get paired-discriminator-cifar10-runs /cifar10-capacity-v1 results/
uv run --extra cifar python -m paired_discriminator.capacity_report
```

Sources: `cifar_capacity.py`, `capacity_eval.py`, `capacity_report.py`,
`modal_capacity.py`. Configurations: `configs/cifar10-d_batch256.json` and
`configs/cifar10-d_width128.json`. Raw tensor checkpoints and neighbor arrays are
retained locally and in Modal; repository ignore rules exclude these large files.

## Execution notes

Both training arms ran entirely on NVIDIA A10 GPUs. Evaluation through 90k
(and batch256 at 100k) used the A10 evaluator. After A10 allocation stalled,
the final wider-D evaluation allowed several GPU types as fallbacks and was
ultimately assigned NVIDIA A10 as well. Its runtime metadata is saved separately. The Modal evaluator now
refreshes its run-volume view before reading newly committed checkpoints,
avoiding stale views in reused containers. No training replay was needed.
