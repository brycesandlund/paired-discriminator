"""Confirm the tuned GAN architecture with uniform D sampling (#34)."""

import json
import time

from .gan_ablation import launch
from .grid_experiment import ROOT


BASE = dict(
    residual_g=True, fourier=True, g_fourier=True, latent=2,
    lr_g=.001, anneal=True, cosine=True, cosine_after=.5,
    alpha=0, power=1,
)
SPECS = [
    dict(BASE, name='paired_uniform', method='paired', seeds=[1, 2, 3, 4]),
    dict(BASE, name='vanilla_uniform', method='vanilla', seeds=[0, 1, 2, 3, 4]),
]


def main():
    # Paired seed0 already used exactly these settings in the prior screen.
    old_specs = json.loads((ROOT / 'results/gan-ablation-r4-v1/specs.json').read_text())
    old = next(s for s in old_specs if s['name'] == 'uniform')
    assert {k: v for k, v in old.items() if k != 'name'} == BASE
    start = time.perf_counter()
    launch('gan-uniform-v1', SPECS, steps=50000, workers=8, eval_samples=10000)
    root = ROOT / 'results/gan-uniform-v1'
    (root / 'launch_metadata.json').write_text(json.dumps({
        'new_training_jobs': 9,
        'reused_paired_seed0': 'results/gan-ablation-r4-v1/uniform_seed0',
        'launch_wall_seconds': time.perf_counter() - start,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
