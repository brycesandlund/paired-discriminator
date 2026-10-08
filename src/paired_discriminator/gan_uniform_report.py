"""Replay and compare uniform versus linear-deficit tuned GANs (#34)."""

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .gan_ablation import make_models
from .grid10_1200k_report import EDGES, metrics
from .grid_experiment import ROOT, centers, sample_real
from .grid_mode_sweep_report import target_grid


ARMS = {
    'vanilla_uniform': 'Vanilla · uniform',
    'paired_uniform': 'Paired · uniform',
    'vanilla_linear': 'Vanilla · linear deficit',
    'paired_linear': 'Paired · linear deficit',
}
METRICS = ['mode_tv', 'spatial_tv', 'valid_fraction', 'coverage_half_target']


def run_path(arm, seed):
    if arm == 'paired_uniform' and seed == 0:
        return ROOT / 'results/gan-ablation-r4-v1/uniform_seed0'
    if arm.endswith('uniform'):
        return ROOT / f'results/gan-uniform-v1/{arm}_seed{seed}'
    if arm == 'vanilla_linear':
        return ROOT / f'results/gan-ablation-vanilla-linear-v1/vanilla_linear_seed{seed}'
    folder = 'gan-ablation-r4-v1' if seed == 0 else 'gan-ablation-validation-v1'
    return ROOT / f'results/{folder}/linear_seed{seed}'


def normalized_spec(spec):
    spec = {k: v for k, v in spec.items() if k not in ('name', 'seeds')}
    spec.setdefault('method', 'paired')
    return spec


def build():
    torch.set_num_threads(1)
    launch_meta = json.loads((ROOT / 'results/gan-uniform-v1/launch_metadata.json').read_text())
    out = ROOT / 'results/gan-uniform-comparison-v1'
    out.mkdir(exist_ok=True)
    roots = {run_path(arm, seed).parent for arm in ARMS for seed in range(5)}
    for root in roots:
        hashes = json.loads((root / 'provenance.json').read_text())['source_sha256']
        for file, expected in hashes.items():
            assert hashlib.sha256((root / 'source' / file).read_bytes()).hexdigest() == expected

    config = json.loads((ROOT / 'results/gan-uniform-v1/config.json').read_text())
    target = target_grid(config, EDGES)
    rows, metadata, replays, rng_checks = [], [], 0, 0
    for seed in range(5):
        checkpoints = {}
        for arm in ARMS:
            path = run_path(arm, seed)
            checkpoint = torch.load(path / 'checkpoint_50000.pt', weights_only=False)
            checkpoints[arm] = checkpoint
            assert checkpoint['seed'] == seed and checkpoint['step'] == 50000
            assert checkpoint['config'] == config
            expected_alpha = 0 if arm.endswith('uniform') else .9
            assert checkpoint['spec']['alpha'] == expected_alpha
            assert checkpoint['spec']['power'] == 1
            assert checkpoint['spec'].get('method', 'paired') == arm.split('_')[0]
            for key in ('optimizer_g', 'optimizer_d'):
                assert all(int(v['step']) == 50000 for v in checkpoint[key]['state'].values())
            meta = json.loads((path / 'metadata.json').read_text())
            assert meta['steps'] == 50000 and meta['seed'] == seed
            metadata.append(dict(arm=arm, path=str(path.relative_to(ROOT)), **meta))
            g, _ = make_models(config, checkpoint['spec'], seed)
            for variant, key in [('raw', 'generator'), ('ema', 'ema')]:
                g.load_state_dict(checkpoint[key])
                z = torch.randn(10000, 2, generator=torch.Generator().manual_seed(70000 + seed))
                with torch.no_grad():
                    assert np.array_equal(g(z).numpy(), np.load(path / f'samples_50000_{variant}.npy'))
                replays += 1
                z = torch.randn(10000, 2, generator=torch.Generator().manual_seed(932000 + seed))
                with torch.no_grad():
                    points = g(z).numpy()
                assert np.isfinite(points).all()
                m, mass = metrics(points, config, target)
                rows.append(dict(arm=arm, seed=seed, variant=variant, accepted_mass=mass.tolist(), **m))
                np.save(out / f'{arm}_{variant}_seed{seed}.npy', points)

        # Uniform methods consume identical real/noise/slot streams. Across
        # samplers only D's real stream may differ (randint versus multinomial).
        for left, right, keys in [
            ('vanilla_uniform', 'paired_uniform', list(checkpoints['vanilla_uniform']['rng'])),
            ('vanilla_linear', 'paired_linear', ['real_g', 'noise_d', 'noise_g', 'slots']),
            ('vanilla_uniform', 'vanilla_linear', ['real_g', 'noise_d', 'noise_g', 'slots']),
            ('paired_uniform', 'paired_linear', ['real_g', 'noise_d', 'noise_g', 'slots']),
        ]:
            a, b = checkpoints[left], checkpoints[right]
            sa, sb = normalized_spec(a['spec']), normalized_spec(b['spec'])
            for spec in (sa, sb):
                spec.pop('alpha')
                spec.pop('method')
            assert sa == sb
            for key in keys:
                assert torch.equal(a['rng'][key], b['rng'][key])
            rng_checks += 1

        points = sample_real(config, 10000, torch.Generator().manual_seed(942000 + seed)).numpy()
        m, mass = metrics(points, config, target)
        rows.append(dict(arm='real', seed=seed, variant='raw', accepted_mass=mass.tolist(), **m))

    summary = []
    for arm in [*ARMS, 'real']:
        for variant in ['raw', 'ema']:
            rr = [r for r in rows if r['arm'] == arm and r['variant'] == variant]
            if not rr:
                continue
            summary.append(dict(
                arm=arm, variant=variant,
                full_coverage_seeds=sum(r['coverage_half_target'] == 100 for r in rr),
                **{k: dict(mean=float(np.mean([r[k] for r in rr])),
                           sd=float(np.std([r[k] for r in rr], ddof=1))) for k in METRICS},
            ))
    for filename, value in [('summary.json', summary), ('metrics.json', rows), ('metadata.json', metadata)]:
        (out / filename).write_text(json.dumps(value, indent=2) + '\n')
    verification = dict(checkpoint_sample_replays=replays, rng_comparison_groups=rng_checks,
                        source_hashes=True, optimizer_counts=True, matched_configs=True,
                        torch=torch.__version__, python=platform.python_version())
    report_source = Path(__file__).read_bytes()
    (out / 'report_source.py').write_bytes(report_source)
    verification['report_source_sha256'] = hashlib.sha256(report_source).hexdigest()
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    mu = centers(config).numpy()
    for ax, arm in zip(axes.flat, ARMS):
        points = np.load(out / f'{arm}_raw_seed0.npy')
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.25)
        ax.scatter(mu[:, 0], mu[:, 1], s=7, c='red')
        ax.set(title=ARMS[arm], xlim=(-7.75, 7.75), ylim=(-7.75, 7.75), aspect='equal')
    fig.suptitle('Tuned GANs · 50k updates · raw weights\n10,000 fixed evaluation samples · seed 0')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / 'samples.png', dpi=160)
    plt.close(fig)

    lines = [
        '# Uniform versus deficit sampling for tuned GANs (#34)', '',
        'All four arms use the frozen successful architecture and optimizer from #33 at 50k updates. '
        'No reconstruction loss. Nine new jobs: four paired seeds plus five vanilla seeds. '
        'Paired-uniform seed0 is reused from its exact #33 screen; all deficit checkpoints are reused. '
        'This confirms uniform sampling under the selected settings, not a separate search optimized for uniform sampling.', '',
        'Raw weights are primary; generator EMA is a separately reported option for every seed. '
        'Five-seed means include development seed0. Evaluation uses the same 10k latent draws per seed '
        'as #33, independent of training and of the saved development-evaluation noise.', '',
        '## Results', '',
        'Mean ± sample SD. Half-target coverage requires at least 0.5% of all samples within '
        'radius .3 of each target center; invalid samples remain in the denominator.', '',
        '| Method | Output | Mode TV ↓ | Fine TV ↓ | Valid mass ↑ | Half-target modes | Full coverage seeds |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in summary:
        def fmt(key):
            return f"{r[key]['mean']:.4f} ± {r[key]['sd']:.4f}"
        lines.append(f"| {ARMS.get(r['arm'], 'Real reference')} | {r['variant']} | "
                     f"{fmt('mode_tv')} | {fmt('spatial_tv')} | {fmt('valid_fraction')} | "
                     f"{r['coverage_half_target']['mean']:.1f}/100 | {r['full_coverage_seeds']}/5 |")
    lines += ['', '## Interpretation', '',
              'The strong feature-based fit survives uniform sampling: both methods retain about 96% valid mass. '
              'Linear deficit lowers mode TV in every matched seed for both objectives and raises full '
              'half-target coverage from one of five to five of five seeds. Uniform raw generators have '
              'slightly lower mean fine TV, so the deficit gain is specifically in mode allocation rather '
              'than a uniform win on distributional metrics. Vanilla-uniform has lower mean mode/fine TV '
              'than paired-uniform, with overlapping seed variation; this gives no evidence of a paired '
              'advantage under these settings. No additional uniform-specific tuning was done.', '',
              '## Individual seeds (raw)', '',
              '| Method | Seed | Mode TV | Fine TV | Valid mass | Half-target modes |',
              '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['variant'] == 'raw' and r['arm'] != 'real':
            lines.append(f"| {ARMS[r['arm']]} | {r['seed']} | {r['mode_tv']:.4f} | "
                         f"{r['spatial_tv']:.4f} | {r['valid_fraction']:.4f} | {r['coverage_half_target']} |")
    lines += ['', '## Budgets and settings', '',
              'G: two width128 LeakyReLU layers, 2D Gaussian latent, generic Fourier features, '
              'and residual output scaled by the shared independent 100k-real-point pilot. '
              'D: two width128 LeakyReLU layers with Fourier features introduced over the first 25k updates. '
              'Adam (.5,.999), lr .001 for 25k then cosine to .0001. Linear deficit alpha=.9,p=1, '
              'mass EMA=.99; uniform alpha=0. G weight EMA=.999.', '',
              'D consumes 12.8M real draws and 12.8M fake draws per run. G consumes 12.8M latent draws. '
              'Paired additionally uses 12.8M uniform G real references; vanilla draws but does not use '
              'these references to preserve RNG parity. Paired has a larger D. Equal updates are not equal compute.', '',
              f"Nine new training jobs finished in {launch_meta['launch_wall_seconds']:.1f} seconds elapsed, "
              'including their training/evaluation and launch overhead, with up to eight concurrent workers.', '',
              'Uniform sampling uses no deficit-derived weights. The existing trainer calculates an unused '
              'mode-mass estimate, but it does not affect sampling, losses, or gradients when alpha=0. '
              'Known mode labels are still used for evaluation. Generic features contain no mode centers or spacing.', '',
              '## Samples', '', 'Seed0 was fixed in advance, not chosen for appearance; raw generators.', '',
              '![Samples](samples.png)', '', '## Verification', '',
              f'{replays} exact checkpoint/sample replays; {rng_checks} matched RNG comparison groups; '
              'all source snapshot hashes, config/spec matches, optimizer counts and completion metadata passed. '
              'Training code is unchanged from #33.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    build()
