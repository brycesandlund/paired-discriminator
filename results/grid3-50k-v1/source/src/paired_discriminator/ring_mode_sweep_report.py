"""Compare mode-count sweeps with a distribution metric that detects filled gaps."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment import ROOT, centers, coverage_metrics

METHODS = ['vanilla', 'rsgan', 'paired']
COLORS = ['#4477AA', '#EEAA33', '#228855']
EDGES = np.linspace(-3, 3, 121)  # Fixed 0.05-wide cells, plus an outside category.


def target_grid(config, edges=EDGES):
    means = centers(config).numpy().astype(float)
    z = (edges[None, :, None] - means[:, None, :]) / (config['sigma'] * math.sqrt(2))
    cdf = .5 * (1 + np.vectorize(math.erf)(z))
    marginal = np.diff(cdf, axis=1)
    grid = np.einsum('ki,kj->ij', marginal[:, :, 0], marginal[:, :, 1]) / config['modes']
    result = np.r_[grid.ravel(), max(0, 1 - grid.sum())]
    return result / result.sum()


def spatial_tv(points, target, edges=EDGES):
    counts = np.histogram2d(points[:, 0], points[:, 1], bins=(edges, edges))[0]
    mass = np.r_[counts.ravel(), len(points) - counts.sum()] / len(points)
    return float(np.abs(mass - target).sum() / 2)


def source(modes, method):
    if modes == 8:
        return ROOT / ('results/ring8-rsgan-50k' if method == 'rsgan' else 'results/ring8-50k')
    return ROOT / f'results/ring{modes}-50k-v1'


def calibration(config, target):
    rng = np.random.default_rng(48192 + config['modes'])
    means = centers(config).numpy()
    scores=[]
    for _ in range(200):
        p = means[rng.integers(len(means), size=10000)] + rng.normal(0, config['sigma'], size=(10000,2))
        scores.append(spatial_tv(p, target))
    # Diagnostic negative control: matching radius + Gaussian noise, but no discrete angular modes.
    theta = rng.uniform(0, 2 * np.pi, 10000)
    smooth = config['radius'] * np.column_stack([np.cos(theta), np.sin(theta)]) + rng.normal(0, config['sigma'], size=(10000,2))
    legacy = coverage_metrics(torch.tensor(smooth, dtype=torch.float32), config)
    return {'real_spatial_tv_mean':float(np.mean(scores)), 'real_spatial_tv_95_interval':np.quantile(scores,[.025,.975]).tolist(),
            'draws':200, 'samples_per_draw':10000, 'seed':48192+config['modes'],
            'continuous_ring_spatial_tv':spatial_tv(smooth,target),
            'continuous_ring_legacy_mode_tv':legacy['mode_tv'],
            'continuous_ring_valid_fraction':legacy['valid_fraction']}


def build(output):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; histories=[]; reference={}; configs={}; hashes={}
    samples={}
    for modes in [8,16,32]:
        config=json.loads((source(modes,'vanilla')/'config.json').read_text());configs[modes]=config
        target=target_grid(config);reference[modes]=calibration(config,target)
        for method in METHODS:
            src=source(modes,method)
            assert json.loads((src/'config.json').read_text())==config
            provenance=json.loads((src/'provenance.json').read_text())
            for name,h in provenance['source_sha256'].items():
                assert hashlib.sha256((src/'source'/name).read_bytes()).hexdigest()==h
            for seed in range(5):
                path=src/f'{method}_seed{seed}'
                with (path/'metrics.csv').open() as f:rr=list(csv.DictReader(f))
                assert len(rr)==51 and int(rr[-1]['step'])==50000
                for r in rr:
                    mass=np.array([float(r[f'mode_{i}_mass']) for i in range(modes)])
                    histories.append({'modes':modes,'method':method,'seed':seed,'step':int(r['step']),
                        'mode_tv':float(r['mode_tv']),'valid_fraction':float(r['valid_fraction']),
                        'coverage_1pct':int(r['coverage']),'coverage_relative':int((mass>=.08/modes).sum())})
                for step in [10000,50000]:
                    name='samples_step10000.npy' if step==10000 else 'final_samples.npy'
                    p=path/name
                    if modes==8 and method!='rsgan' and step==10000:
                        p=ROOT/'results/ring8-v1'/f'{method}_seed{seed}'/'final_samples.npy'
                    points=np.load(p);assert points.shape==(10000,2) and np.isfinite(points).all()
                    hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    actual=coverage_metrics(torch.tensor(points),config)
                    r=next(r for r in rr if int(r['step'])==step)
                    for metric,value in actual.items():assert abs(value-float(r[metric]))<1e-10
                    mass=np.array([actual[f'mode_{i}_mass'] for i in range(modes)])
                    row={'modes':modes,'method':method,'seed':seed,'step':step,
                         'mode_tv':actual['mode_tv'],'valid_fraction':actual['valid_fraction'],
                         'coverage_1pct':actual['coverage'],'coverage_relative':int((mass>=.08/modes).sum()),
                         'spatial_tv':spatial_tv(points,target),'training_seconds':float(r['training_seconds'])}
                    rows.append(row);samples[(modes,method,seed,step)]=(points,mass)
    for name,rr in [('endpoints.csv',rows),('learning_curves.csv',histories)]:
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rr[0]);writer.writeheader();writer.writerows(rr)
    summary=[]
    for modes in [8,16,32]:
        for step in [10000,50000]:
            for method in METHODS:
                selected=[r for r in rows if r['modes']==modes and r['step']==step and r['method']==method]
                stats={'modes':modes,'step':step,'method':method}
                for metric in ['mode_tv','spatial_tv','valid_fraction','coverage_1pct','coverage_relative']:
                    vals=[r[metric] for r in selected]
                    stats[metric]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))}
                stats['full_relative_coverage_seeds']=sum(r['coverage_relative']==modes for r in selected)
                summary.append(stats)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'reference_calibration.json').write_text(json.dumps(reference,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'verified_endpoints':len(rows),'training_trajectories':45,'samples_sha256':hashes},indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,step in enumerate([10000,50000]):
        for j,(metric,title) in enumerate([('mode_tv','Legacy mode TV ↓'),('spatial_tv','Fine-grid distribution TV ↓'),('coverage_relative','Fraction of modes covered ↑')]):
            ax=axes[i,j]
            for method,color in zip(METHODS,COLORS):
                selected=[next(r for r in summary if r['modes']==m and r['step']==step and r['method']==method) for m in [8,16,32]]
                denom=np.array([8,16,32]) if metric=='coverage_relative' else 1
                mean=np.array([r[metric]['mean'] for r in selected])/denom
                sd=np.array([r[metric]['sd'] for r in selected])/denom
                ax.errorbar([8,16,32],mean,yerr=sd,marker='o',capsize=3,label=method,color=color)
            if metric=='spatial_tv':
                ax.plot([8,16,32],[reference[m]['real_spatial_tv_mean'] for m in [8,16,32]],'k--',label='real samples (finite-n reference)')
            ax.set_xticks([8,16,32]);ax.set_xlabel('Gaussian components');ax.set_title(f'{step//1000}k updates · {title}');ax.grid(alpha=.2)
            if metric=='coverage_relative':ax.set_ylim(0,1.05)
            else:ax.set_ylim(bottom=0)
    axes[0,0].legend();axes[0,1].legend(fontsize=8)
    fig.suptitle('Fixed radius 2, sigma 0.1 · five seeds · mean ± sample SD\nCoverage: accepted mass ≥8% of each mode’s target mass')
    fig.tight_layout();fig.savefig(output/'mode_count_comparison.png',dpi=160);plt.close(fig)
    for step in [10000,50000]:
        fig,axes=plt.subplots(9,5,figsize=(13,21),sharex=True,sharey=True)
        for i,modes in enumerate([8,16,32]):
            ctr=centers(configs[modes]).numpy()
            for j,method in enumerate(METHODS):
                for seed in range(5):
                    ax=axes[3*i+j,seed];points,_=samples[(modes,method,seed,step)]
                    ax.scatter(points[:3000,0],points[:3000,1],s=.8,alpha=.35,color=COLORS[j],rasterized=True)
                    ax.scatter(ctr[:,0],ctr[:,1],s=10,marker='x',color='black');ax.set_aspect('equal');ax.set_xlim(-3,3);ax.set_ylim(-3,3)
                    if seed==0:ax.set_ylabel(f'{modes} modes\n{method}')
                    if i==j==0:ax.set_title(f'Seed {seed}')
        fig.suptitle(f'{step//1000}k updates · first 3,000 fixed generated samples · × = component means')
        fig.tight_layout();fig.savefig(output/f'samples_{step//1000}k.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,3,figsize=(15,10))
        for i,modes in enumerate([8,16,32]):
            for j,method in enumerate(METHODS):
                ax=axes[i,j];mass=np.array([samples[(modes,method,s,step)][1] for s in range(5)])
                ax.bar(range(modes),mass.mean(0)*modes,color=COLORS[j]);ax.errorbar(range(modes),mass.mean(0)*modes,yerr=mass.std(0,ddof=1)*modes,fmt='none',ecolor='black',lw=.7)
                ax.axhline(1,color='black',ls='--');ax.set_title(f'{modes} modes · {method}');ax.set_xlabel('Mode index');ax.set_ylabel('Accepted mass / target mass')
        fig.suptitle(f'{step//1000}k updates · mean ± sample SD across five seeds');fig.tight_layout();fig.savefig(output/f'accepted_mass_{step//1000}k.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(14,9),sharex=True,sharey=True)
    for i,modes in enumerate([8,16,32]):
        config=configs[modes];ctr=centers(config).numpy();rng=np.random.default_rng(59100+modes)
        real=ctr[rng.integers(modes,size=3000)]+rng.normal(0,.1,(3000,2))
        theta=rng.uniform(0,2*np.pi,3000)
        smooth=2*np.column_stack([np.cos(theta),np.sin(theta)])+rng.normal(0,.1,(3000,2))
        points_list=[real]+[samples[(modes,m,0,50000)][0][:3000] for m in METHODS]+[smooth]
        for j,(points,title,color) in enumerate(zip(points_list,['True mixture','Vanilla, seed 0','RSGAN, seed 0','Paired, seed 0','Smooth-ring control'],['#666666']+COLORS+['#884488'])):
            ax=axes[i,j];ax.scatter(points[:,0],points[:,1],s=.8,alpha=.4,color=color,rasterized=True)
            ax.scatter(ctr[:,0],ctr[:,1],marker='x',s=8,color='black');ax.set_aspect('equal');ax.set_xlim(-2.7,2.7);ax.set_ylim(-2.7,2.7)
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(f'{modes} components')
    fig.suptitle('Fixed geometry · true distribution versus 50k samples and smooth-ring control')
    fig.tight_layout();fig.savefig(output/'reference_and_samples.png',dpi=160);plt.close(fig)
    sensitivity=[]
    for width in [.025,.05,.1]:
        edges=np.linspace(-3,3,round(6/width)+1)
        for modes in [8,16,32]:
            target=target_grid(configs[modes],edges)
            for step in [10000,50000]:
                for method in METHODS:
                    values=[spatial_tv(samples[(modes,method,seed,step)][0],target,edges) for seed in range(5)]
                    sensitivity.append({'cell_width':width,'modes':modes,'step':step,'method':method,'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))})
    (output/'grid_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    lines=['# Ring mode-count sweep','','8, 16 and 32 equally weighted isotropic Gaussian components on a radius-2 ring, sigma 0.1 per axis. Original vanilla, RSGAN and paired objectives/networks; five seeds, 50k updates with retained 10k draws. Existing 8-mode runs reused unchanged; 30 new runs for 16/32 modes.','',
      '## Findings','',
      'At 16 components, all methods cover all modes in all five seeds at both endpoints, even with the original 1% cutoff. Paired has lower average legacy mode TV at 50k (0.088 versus 0.171 vanilla and 0.165 RSGAN), but worse fine-grid TV (0.568 versus 0.539 and 0.534). At 10k paired is worse on both mean metrics than the baselines.', '',
      'At 32 components, every method covers all modes at both endpoints. At 50k paired has slightly worse legacy mode TV (0.0535 versus 0.0381 vanilla and 0.0347 RSGAN), but lower mean fine-grid TV (0.526 versus 0.581 and 0.554). All remain far from the real-sample fine-grid reference of 0.192. Samples show thin rings and imperfect component structure, so full reported coverage is not evidence that all Gaussian densities have been reproduced.', '',
      'The new fine-grid audit also qualifies the original 8-mode result: at 50k paired has clearly better coarse mode balance, but fine-grid TV is similar across methods (0.555 paired, 0.568 vanilla, 0.540 RSGAN; real reference 0.111). The 8-mode coverage finding remains valid; it is not a full-density fidelity result. Mean fine-grid TV worsens from 10k to 50k in every method/mode-count combination at the primary 0.05 cell width. Thin or shifted clusters, bridges and insufficient Gaussian spread are visible in the plots.', '',
      'Adding components at fixed radius changes the geometry enough that it is not monotonically harder on coarse coverage. A future separation-preserving sweep would address a different question and would need to control for the accompanying change in data scale or Gaussian width. No such additional training is included here. Grid-width sensitivity at 0.025, 0.05 and 0.1 is retained in grid_sensitivity.json.', '',
      '## Protocol and interpretation limits','',
      'Only component count changes. Increasing count also reduces separation and samples per component at fixed batch size: this is a fixed-geometry stress test, not an isolated causal test of number of modes. Adjacent means are 1.531, 0.780 and 0.392 apart (15.31, 7.80 and 3.92 sigma). Networks, learning rate, 256-sample batches, 1:1 updates, latent dimension and evaluation draws are unchanged. CPU training, one thread per run; four independent workers. Runtime is not a matched-compute benchmark against historical runs.','',
      'Legacy valid mass is assigned to the nearest mean when within 3 sigma; legacy mode TV includes an invalid category with target zero. The old 1%-of-all-samples coverage cutoff is retained, but corresponds to 8%, 16% and 32% of a component’s target mass as count increases. The main coverage comparison instead fixes this fraction at 8% of target mass (cutoffs 1%, 0.5%, 0.25%). Coverage remains a coarse threshold and should not replace distributional metrics.','',
      'At 32 components the 3-sigma acceptance disks overlap. A smooth ring may score well without producing the specified Gaussian density. Fine-grid TV therefore bins all samples into fixed 0.05×0.05 cells on [-3,3]² plus an outside category, and compares their mass with exact Gaussian-mixture cell probabilities computed via the normal CDF. It includes spatial shape within and between components. The same grid applies to every mode count. Finite samples make its ideal-distribution score nonzero: 200 independent real draws of 10,000 points provide the reference mean and central 95% interval. This interval describes real-sample variability, not training uncertainty.','',
      '## Endpoint results','','Mean ± sample SD across five training seeds. Relative coverage uses ≥8% of target mass.','',
      '| Modes | Updates | Method | Legacy mode TV ↓ | Fine-grid TV ↓ | Relative coverage | Full-coverage seeds |','|---:|---:|---|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
        lines.append(f"| {r['modes']} | {r['step']} | {r['method']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['coverage_relative']['mean']:.1f}/{r['modes']} | {r['full_relative_coverage_seeds']}/5 |")
    lines+=['','![Mode-count comparison](mode_count_comparison.png)','','## Real-distribution and smooth-ring controls','','| Modes | Real fine-grid TV mean | Real central 95% | Smooth-ring fine-grid TV | Smooth-ring legacy TV |','|---:|---:|---|---:|---:|']
    for m,r in reference.items():
        lines.append(f"| {m} | {r['real_spatial_tv_mean']:.4f} | {r['real_spatial_tv_95_interval']} | {r['continuous_ring_spatial_tv']:.4f} | {r['continuous_ring_legacy_mode_tv']:.4f} |")
    lines+=['','## Samples and accepted mass','','![True mixture and sample comparison](reference_and_samples.png)','','- [10k samples, every seed](samples_10k.png)','- [50k samples, every seed](samples_50k.png)','- [10k accepted mass](accepted_mass_10k.png)','- [50k accepted mass](accepted_mass_50k.png)','','## Reproduce','','Run `uv run python -m paired_discriminator.ring_mode_sweep` in a fresh output version; it refuses to overwrite existing results. Build this report with `uv run python -m paired_discriminator.ring_mode_sweep_report results/ring-mode-count-v1`. Configurations: `configs/ring16-50k.json` and `configs/ring32-50k.json`. Source snapshots and sample hashes are verified; all 90 endpoint metrics are recomputed from saved samples. No GAN architecture or objective changes.','']
    (output/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    import sys
    build(sys.argv[1])
