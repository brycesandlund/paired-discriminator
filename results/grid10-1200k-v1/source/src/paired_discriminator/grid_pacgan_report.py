"""Compare PacGAN2 with saved grid baselines; validate every reported endpoint."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_pacgan import ROOT, models
from .grid_experiment import centers, coverage_metrics
from .grid_mode_sweep_report import target_grid, spatial_tv

METHODS = ['vanilla', 'rsgan', 'paired', 'pacgan2']
COLORS = ['#4477AA', '#EEAA33', '#228855', '#AA3377']
METRICS = ['mode_tv', 'spatial_tv', 'valid_fraction', 'coverage_1pct', 'coverage_relative']


def run_path(side, method, step):
    suffix = 'pacgan2' if method == 'pacgan2' else ('150k' if step > 50000 else '50k')
    return ROOT / f'results/grid{side}-{suffix}-v1'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, histories, samples, configs, hashes = [], [], {}, {}, {}
    checked_sources = set()
    replayed = 0
    matched_rng_checks = 0
    reference = json.loads((ROOT / 'results/grid-mode-count-v1/reference_calibration.json').read_text())
    for side in (3, 5, 7):
        modes = side * side
        config = json.loads((ROOT / f'results/grid{side}-50k-v1/config.json').read_text())
        configs[side] = config
        target = target_grid(config)
        steps = [10000, 50000] + ([100000, 150000] if side == 7 else [])
        for method in METHODS:
            for seed in range(5):
                # Merge baseline continuation logs, retaining the original boundary row.
                all_history = {}
                for src in sorted({run_path(side, method, step) for step in steps}):
                    run_config = json.loads((src / 'config.json').read_text())
                    assert {k:v for k,v in run_config.items() if k != 'steps'} == {k:v for k,v in config.items() if k != 'steps'}
                    if src not in checked_sources:
                        provenance = json.loads((src / 'provenance.json').read_text())
                        for name, digest in provenance['source_sha256'].items():
                            assert hashlib.sha256((src / 'source' / name).read_bytes()).hexdigest() == digest
                        checked_sources.add(src)
                    path = src / f'{method}_seed{seed}'
                    with (path / 'metrics.csv').open() as handle:
                        history = list(csv.DictReader(handle))
                    assert int(history[-1]['step']) == run_config['steps']
                    for r in history:
                        step = int(r['step'])
                        if step not in all_history or src.name.endswith('50k-v1'):
                            all_history[step] = r
                assert sorted(all_history) == list(range(0, steps[-1] + 1, 1000))
                for step, r in sorted(all_history.items()):
                    mass = np.array([float(r[f'mode_{i}_mass']) for i in range(modes)])
                    histories.append(dict(modes=modes, method=method, seed=seed, step=step,
                        mode_tv=float(r['mode_tv']), valid_fraction=float(r['valid_fraction']),
                        coverage_1pct=int(r['coverage']), coverage_relative=int((mass >= .08/modes).sum())))
                for step in steps:
                    path = run_path(side, method, step) / f'{method}_seed{seed}'
                    if method == 'pacgan2':
                        name = f'samples_step{step}.npy'
                        checkpoint = path / f'checkpoint_{step}.pt'
                    else:
                        name = f'samples_step{step}.npy' if step in (10000, 100000) else 'final_samples.npy'
                        checkpoint = path / ('checkpoint_100000.pt' if step == 100000 else 'checkpoint.pt')
                    sample_path = path / name
                    points = np.load(sample_path)
                    assert points.shape == (10000, 2) and np.isfinite(points).all()
                    hashes[str(sample_path.relative_to(ROOT))] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
                    actual = coverage_metrics(torch.from_numpy(points), config)
                    r = all_history[step]
                    for key, value in actual.items():
                        assert abs(value - float(r[key])) < 1e-10, (path, step, key)
                    # Original 10k checkpoints were not retained. All other endpoints replay.
                    if method == 'pacgan2' or step != 10000:
                        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                        assert saved['step'] == step and saved['seed'] == seed and saved['method'] == method
                        g, _ = models(config, method, seed)
                        g.load_state_dict(saved['generator'])
                        z = torch.randn(10000, config['latent_dim'], generator=torch.Generator().manual_seed(seed + 70000))
                        with torch.no_grad():
                            assert np.array_equal(g(z).numpy(), points)
                        for optimizer in ('optimizer_g', 'optimizer_d'):
                            assert all(int(state['step']) == step for state in saved[optimizer]['state'].values())
                        if method == 'pacgan2' and step != 10000:
                            paired_path = run_path(side, 'paired', step) / f'paired_seed{seed}'
                            paired_checkpoint = paired_path / ('checkpoint_100000.pt' if step == 100000 else 'checkpoint.pt')
                            paired_state = torch.load(paired_checkpoint, map_location='cpu', weights_only=False)
                            for key, value in saved['rng_states'].items():
                                assert torch.equal(value, paired_state['rng_states'][key]), (side, seed, step, key)
                            matched_rng_checks += 1
                        replayed += 1
                    mass = np.array([actual[f'mode_{i}_mass'] for i in range(modes)])
                    rows.append(dict(modes=modes, method=method, seed=seed, step=step,
                        mode_tv=actual['mode_tv'], spatial_tv=spatial_tv(points, target),
                        valid_fraction=actual['valid_fraction'], coverage_1pct=actual['coverage'],
                        coverage_relative=int((mass >= .08/modes).sum()), source=str(sample_path.relative_to(ROOT))))
                    samples[(side, method, seed, step)] = (points, mass)
    assert len(rows) == 160 and replayed == 115 and matched_rng_checks == 25
    write_csv(output / 'endpoints.csv', rows)
    write_csv(output / 'learning_curves.csv', histories)
    summary = []
    comparisons = []
    for side in (3, 5, 7):
        for step in [10000, 50000] + ([100000, 150000] if side == 7 else []):
            for method in METHODS:
                selected = [r for r in rows if (r['modes'], r['step'], r['method']) == (side*side, step, method)]
                stats = dict(modes=side*side, step=step, method=method)
                for metric in METRICS:
                    values = [r[metric] for r in selected]
                    stats[metric] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
                stats['full_1pct_seeds'] = sum(r['coverage_1pct'] == side*side for r in selected)
                stats['full_relative_seeds'] = sum(r['coverage_relative'] == side*side for r in selected)
                summary.append(stats)
            for other in METHODS[:-1]:
                for metric in ('mode_tv', 'spatial_tv'):
                    difference = []
                    for seed in range(5):
                        get = lambda m: next(r[metric] for r in rows if (r['modes'],r['step'],r['method'],r['seed']) == (side*side,step,m,seed))
                        difference.append(get('pacgan2') - get(other))
                    comparisons.append(dict(modes=side*side,step=step,metric=metric,against=other,
                        pacgan2_wins=sum(x<0 for x in difference),differences=difference,
                        mean_difference=float(np.mean(difference))))
    for name, data in [('summary.json',summary), ('paired_comparisons.json',comparisons),
                       ('reference_calibration.json',reference),
                       ('verification.json',dict(verified_endpoints=160, checkpoint_replays=replayed, matched_rng_checks=matched_rng_checks,
                        original_10k_samples_without_checkpoint=45, new_trajectories=15,
                        reused_trajectories=45, samples_sha256=hashes))]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for j,side in enumerate((3,5,7)):
        for i,metric in enumerate(('mode_tv','valid_fraction')):
            ax = axes[i,j]
            for method,color in zip(METHODS,COLORS):
                trajectories = []
                for seed in range(5):
                    rr = [r for r in histories if (r['modes'],r['method'],r['seed']) == (side*side,method,seed)]
                    x = np.array([r['step'] for r in rr])/1000
                    y = np.array([r[metric] for r in rr]); trajectories.append(y)
                    ax.plot(x,y,color=color,alpha=.15,lw=.6)
                values = np.array(trajectories)
                mean, sd = values.mean(0), values.std(0,ddof=1)
                ax.plot(x,mean,label=method,color=color)
                ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1)
            ax.set_title(f'{side}×{side} · ' + ('Mode TV ↓' if i==0 else 'Valid fraction ↑'))
            ax.set_ylim(0,1);ax.set_xlabel('Updates (thousands)');ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('PacGAN2 versus saved baselines · five seeds · mean ± sample SD\nShared G; PacGAN2 and paired share D architecture and D sample workload')
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for side,ax in zip((3,5,7),axes):
        steps = [10000,50000]+([100000,150000] if side==7 else [])
        for method,color in zip(METHODS,COLORS):
            rr=[next(r for r in summary if (r['modes'],r['method'],r['step'])==(side*side,method,step)) for step in steps]
            ax.errorbar(np.array(steps)/1000,[r['spatial_tv']['mean'] for r in rr],yerr=[r['spatial_tv']['sd'] for r in rr],color=color,label=method,marker='o',capsize=3)
        ref=reference[str(side*side)]
        ax.axhline(ref['real_spatial_tv_mean'],color='black',ls='--',label='Real-sample reference')
        ax.axhspan(*ref['real_spatial_tv_95_interval'],color='black',alpha=.08)
        ax.set_title(f'{side}×{side} · Fine-grid TV ↓');ax.set_xlabel('Updates (thousands)');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Density fit · 0.05-wide cells · 10,000 samples per endpoint')
    fig.tight_layout();fig.savefig(output/'density_tv.png',dpi=160);plt.close(fig)
    # Every seed is visible, avoiding selection based on final performance.
    for side in (3,5,7):
        for step in [10000,50000]+([100000,150000] if side==7 else []):
            fig,axes=plt.subplots(4,5,figsize=(13,10.5),sharex=True,sharey=True)
            ctr=centers(configs[side]).numpy()
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                for seed in range(5):
                    ax=axes[j,seed];points,mass=samples[(side,method,seed,step)]
                    ax.scatter(points[:3000,0],points[:3000,1],s=.7,alpha=.35,color=color,rasterized=True)
                    ax.scatter(ctr[:,0],ctr[:,1],marker='x',s=8,color='black')
                    limit=(side-1)/2*1.5+.7
                    ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal')
                    if seed==0:ax.set_ylabel(method)
                    if j==0:ax.set_title(f'Seed {seed}')
            fig.suptitle(f'{side}×{side} · {step//1000}k updates · 3,000 fixed samples per panel · × = mode centers')
            fig.tight_layout();fig.savefig(output/f'samples_grid{side}_{step//1000}k.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,10),layout='constrained')
    for i,side in enumerate((3,5,7)):
        step=150000 if side==7 else 50000
        for j,method in enumerate(METHODS):
            ratio=np.mean([samples[(side,method,seed,step)][1] for seed in range(5)],axis=0)*side*side
            ax=axes[i,j]
            im=ax.imshow(np.log2(np.clip(ratio,1/16,16)).reshape(side,side),origin='lower',cmap='RdBu_r',vmin=-4,vmax=4)
            ax.set_title(f'{side}×{side} · {step//1000}k · {method}')
            coords=(np.arange(side)-(side-1)/2)*1.5
            ax.set_xticks(range(side),[f'{x:g}' for x in coords],fontsize=8)
            ax.set_yticks(range(side),[f'{x:g}' for x in coords],fontsize=8)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,ticks=[-4,-2,0,2,4])
    cb.ax.set_yticklabels(['≤1/16×','1/4×','1× target','4×','≥16×'])
    fig.suptitle('Final accepted mass / ideal mode mass · five-seed mean · log scale')
    fig.savefig(output/'mode_mass_maps.png',dpi=160);plt.close(fig)
    # Original 1%-coverage can hide weak mass; expose the entire mass allocation.
    fig,axes=plt.subplots(3,4,figsize=(17,10))
    for i,side in enumerate((3,5,7)):
        step=150000 if side==7 else 50000
        for j,(method,color) in enumerate(zip(METHODS,COLORS)):
            ax=axes[i,j]
            ratios=np.array([samples[(side,method,seed,step)][1] for seed in range(5)])*side*side
            ax.bar(range(side*side),ratios.mean(0),color=color)
            ax.errorbar(range(side*side),ratios.mean(0),yerr=ratios.std(0,ddof=1),fmt='none',ecolor='black',lw=.6)
            ax.axhline(1,color='black',ls='--',lw=.8)
            ax.set_title(f'{side}×{side} · {step//1000}k · {method}')
            ax.set_xlabel('Mode index');ax.set_ylabel('Accepted mass / target')
    fig.suptitle('Final per-mode accepted mass · mean ± sample SD across five seeds')
    fig.tight_layout();fig.savefig(output/'accepted_mass.png',dpi=160);plt.close(fig)
    write_report(output,summary,comparisons)
    files=['grid_pacgan.py','grid_pacgan_run.py','grid_pacgan_report.py','grid_experiment.py','grid_mode_sweep_report.py']
    source=output/'source';source.mkdir(exist_ok=True)
    provenance={}
    for name in files:
        path=ROOT/'src/paired_discriminator'/name
        shutil.copyfile(path,source/name)
        provenance[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def write_report(output,summary,comparisons):
    lines=['# PacGAN2 on fixed-spacing Gaussian grids','',
        'Fifteen new PacGAN2 trajectories: seeds 0–4 on 3×3 and 5×5 through 50k, and 7×7 through 150k. Reuse all vanilla, RSGAN and paired baselines from experiments #15–16. Evaluate 10k/50k for every grid and additionally 100k/150k for 7×7. Endpoints on a trajectory are not independent trials.','',
        '## Findings','',
        'PacGAN2 closely tracks paired’s learning curves, including the slower early progress on larger grids. Both strongly improve final mode allocation over vanilla and RSGAN. There is no consistent overall winner between PacGAN2 and paired across these metrics and budgets.\n\nAt 50k on 3×3, mode TV is 0.064 ± 0.010 PacGAN2 versus 0.058 ± 0.006 paired; both cover all nine modes under both thresholds in every seed. Fine-grid TV slightly favors PacGAN2 on average (0.599 versus 0.621), with mixed seed-level outcomes. On 5×5, PacGAN2 improves mode TV (0.216 versus 0.242, winning 4/5 matched seeds), while paired improves fine-grid TV (0.597 versus 0.622, winning 4/5). Both recover all 25 modes at the relative threshold in every seed, and at the original 1% threshold in four of five seeds.\n\nOn 7×7, PacGAN2 also starts behind the unary baselines, then overtakes them with more training. At 150k its mode TV is 0.229 ± 0.042 versus paired’s 0.231 ± 0.045, vanilla’s 0.520 and RSGAN’s 0.519. The near-equal mean hides seed variation: paired has lower mode TV in 4/5 matched seeds, while PacGAN2’s larger win on seed 0 offsets those differences. Fine-grid TV is 0.655 PacGAN2 versus 0.647 paired, both far above the true-mixture sampling reference of 0.264. Original-threshold coverage averages 43.2/49 versus 41.4/49; each reaches all 49 in only one seed. Samples still show narrow clusters and bridges.\n\nInterpretation: the gains observed here do not require our real–fake reference arrangement; same-source packing produces very similar improvements with the same joint D architecture. This does not establish an identical mechanism or rule out benefits on other tasks. It makes PacGAN2 a necessary baseline for subsequent claims. No method-specific tuning was performed, and the G-step discriminator workload is smaller for PacGAN2.','',
        '## Method and workload','',
        '[PacGAN](https://arxiv.org/pdf/1712.04086) with pack size 2, applied to our existing BCE GAN; this is not a reproduction of the paper’s architecture or dataset settings. Each pack contains two independent draws from the same source (real or generated), without mode matching. D predicts whether a pack is real. G uses non-saturating BCE targeting real, differentiating through both members.','',
        '| Per training step | Vanilla | RSGAN | Paired | PacGAN2 |',
        '|---|---:|---:|---:|---:|',
        '| D real / fake points | 256 / 256 | 256 / 256 | 256 / 256 | 256 / 256 |',
        '| D network input rows | 512 unary | 512 unary | 256 mixed pairs | 256 same-source packs |',
        '| D BCE decisions | 512 | 256 differences | 256 | 256 |',
        '| G generated points with gradients | 256 | 256 | 256 | 256 |',
        '| D input rows during G update | 256 unary | 512 unary | 256 mixed pairs | 128 fake packs |',
        '| G real references used | 0 | 256 | 256 | 0 |',
        '| D parameters | 17,025 | 17,025 | 17,281 | 17,281 |','',
        'Every step has one D optimizer update and one G optimizer update; an additional 256 fake points are generated without gradients for D. All methods share G (16→128→128→2, 18,946 parameters), Adam lr 0.0002, betas (0.5,0.999), batch 256 and independent seed-specific sampling streams. Paired and PacGAN2 share the exact 4→128→128→1 D architecture and initialization. Hidden activations are LeakyReLU(0.2). PacGAN2 D averages BCE over 128 real-real and 128 fake-fake packs; G averages over 128 fake-fake packs. Real-G and slot draws are retained but unused for PacGAN2. The G-step D workload differs; this is not exactly matched total compute.','',
        'Grid spacing 1.5, sigma 0.1, no coordinate normalization or class labels. Four local CPU workers with one torch thread each. Fixed 10,000-point evaluation noise per seed; coarse metrics every 1k; sample arrays and full checkpoints at requested endpoints.','',
        '## Results','',
        'Mean ± sample SD across five seeds. Mode TV penalizes accepted mass imbalance and invalid points. Validity uses radius 0.3; Gaussian tails mean the true distribution does not have perfect validity. Coverage counts modes receiving ≥1% of all samples, or ≥8% of ideal 1/K mass (the much more lenient relative threshold). Fine-grid TV uses exact mixture probabilities in 0.05-wide cells over [-5.5,5.5]² plus an outside category. The real-sample reference accounts for finite-sample noise and is not a training uncertainty interval.','',
        '| Modes | Updates | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid | Coverage ≥1% | Relative coverage |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
        lines.append(f"| {r['modes']} | {r['step']//1000}k | {r['method']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['valid_fraction']['mean']:.1%} | {r['coverage_1pct']['mean']:.1f}/{r['modes']} | {r['coverage_relative']['mean']:.1f}/{r['modes']} |")
    lines+=['','![Learning curves](training_curves.png)','','![Density TV](density_tv.png)','','![Mode mass maps](mode_mass_maps.png)','','![Accepted mass](accepted_mass.png)','','## Matched-seed comparisons','','Negative differences favor PacGAN2. Seeds pair initialization and random streams, not identical trajectories.','','| Modes | Updates | Metric | Against | PacGAN2 wins | Mean PacGAN2 − other |','|---:|---:|---|---|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['modes']} | {r['step']//1000}k | {r['metric']} | {r['against']} | {r['pacgan2_wins']}/5 | {r['mean_difference']:+.4f} |")
    lines+=['','## All-seed sample panels','']
    for side in (3,5,7):
        for step in [10000,50000]+([100000,150000] if side==7 else []):
            lines.append(f'- [{side}×{side}, {step//1000}k](samples_grid{side}_{step//1000}k.png)')
    lines+=['','## Verification and reproduction','',
        'All 160 endpoint metrics were recomputed from saved samples and checked against logs. All source snapshot hashes and configs were checked. 115 endpoints, including all 40 PacGAN2 endpoints, regenerate bit-for-bit from checkpoints; the original 45 baseline 10k sample arrays have no retained matching checkpoint. Optimizer step counters match every replayed endpoint. All six RNG states match paired at each of the 25 mutually retained checkpoints, confirming equal sampling-stream consumption. Every trajectory has its full expected 1k evaluation sequence. Sample hashes and analysis source are retained. Existing baseline files were read without modification.','',
        'Run `uv run python -m paired_discriminator.grid_pacgan_run` with fresh output directories, then `uv run python -m paired_discriminator.grid_pacgan_report results/grid-pacgan2-comparison-v1`. The launcher refuses overwrites. Configs reuse `grid3-50k.json`, `grid5-50k.json`, `grid7-150k.json`. Tests verify identical initialization, exact packing/sample counts, D fake detachment and gradients through both G pack members.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    import sys
    build(sys.argv[1])
