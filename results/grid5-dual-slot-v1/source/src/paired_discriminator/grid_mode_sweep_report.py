"""Report fixed-spacing Gaussian grids across methods, seeds and training budgets."""
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
from .grid_experiment import ROOT, centers, coverage_metrics

METHODS = ['vanilla', 'rsgan', 'paired']
COLORS = ['#4477AA', '#EEAA33', '#228855']
EDGES = np.linspace(-5.5, 5.5, 221)  # Fixed 0.05-wide cells, plus an outside category.


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
    return ROOT / f'results/grid{math.isqrt(modes)}-50k-v1'


def calibration(config, target):
    rng = np.random.default_rng(48192 + config['modes'])
    means = centers(config).numpy()
    scores=[]
    for _ in range(200):
        p = means[rng.integers(len(means), size=10000)] + rng.normal(0, config['sigma'], size=(10000,2))
        scores.append(spatial_tv(p, target))
    return {'real_spatial_tv_mean':float(np.mean(scores)), 'real_spatial_tv_95_interval':np.quantile(scores,[.025,.975]).tolist(),
            'draws':200, 'samples_per_draw':10000, 'seed':48192+config['modes']}


def build(output):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; histories=[]; reference={}; configs={}; hashes={}
    samples={}
    for modes in [9,25,49]:
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
    for modes in [9,25,49]:
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
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for i,step in enumerate([10000,50000]):
        for j,(metric,title) in enumerate([('mode_tv','Mode TV ↓'),('spatial_tv','Fine-grid distribution TV ↓'),('coverage_relative','Fraction of modes covered ↑'),('valid_fraction','Valid sample fraction ↑')]):
            ax=axes[i,j]
            for method,color in zip(METHODS,COLORS):
                selected=[next(r for r in summary if r['modes']==m and r['step']==step and r['method']==method) for m in [9,25,49]]
                denom=np.array([9,25,49]) if metric=='coverage_relative' else 1
                mean=np.array([r[metric]['mean'] for r in selected])/denom
                sd=np.array([r[metric]['sd'] for r in selected])/denom
                ax.errorbar([9,25,49],mean,yerr=sd,marker='o',capsize=3,label=method,color=color)
            if metric=='spatial_tv':
                ax.plot([9,25,49],[reference[m]['real_spatial_tv_mean'] for m in [9,25,49]],'k--',label='real samples (finite-n reference)')
            ax.set_xticks([9,25,49]);ax.set_xlabel('Gaussian components');ax.set_title(f'{step//1000}k updates · {title}');ax.grid(alpha=.2)
            if metric in ['coverage_relative','valid_fraction']:ax.set_ylim(0,1.05)
            else:ax.set_ylim(bottom=0)
    axes[0,0].legend();axes[0,1].legend(fontsize=8)
    fig.suptitle('Grid spacing 1.5, sigma 0.1 · five seeds · mean ± sample SD\nCoverage: accepted mass ≥8% of each mode’s target mass')
    fig.tight_layout();fig.savefig(output/'mode_count_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for j,modes in enumerate([9,25,49]):
        for i,(metric,title) in enumerate([('mode_tv','Mode TV ↓'),('valid_fraction','Valid sample fraction ↑')]):
            ax=axes[i,j]
            for method,color in zip(METHODS,COLORS):
                trajectories=[]
                for seed in range(5):
                    rr=sorted([r for r in histories if r['modes']==modes and r['method']==method and r['seed']==seed],key=lambda r:r['step'])
                    x=np.array([r['step'] for r in rr])/1000
                    values=np.array([r[metric] for r in rr]);trajectories.append(values)
                    ax.plot(x,values,color=color,alpha=.15,lw=.7)
                a=np.array(trajectories);mean=a.mean(0);sd=a.std(0,ddof=1)
                ax.plot(x,mean,color=color,label=method)
                ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1)
            ax.set_ylim(0,1);ax.set_title(f'{math.isqrt(modes)}×{math.isqrt(modes)} · {title}');ax.set_xlabel('Updates (thousands)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Fixed-spacing grids · five seeds · thin seed trajectories, bold mean, shaded ±SD')
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    for step in [10000,50000]:
        fig,axes=plt.subplots(9,5,figsize=(13,21),sharex=True,sharey=True)
        for i,modes in enumerate([9,25,49]):
            ctr=centers(configs[modes]).numpy()
            for j,method in enumerate(METHODS):
                for seed in range(5):
                    ax=axes[3*i+j,seed];points,_=samples[(modes,method,seed,step)]
                    ax.scatter(points[:3000,0],points[:3000,1],s=.8,alpha=.35,color=COLORS[j],rasterized=True)
                    ax.scatter(ctr[:,0],ctr[:,1],s=10,marker='x',color='black');ax.set_aspect('equal');ax.set_xlim(-5.5,5.5);ax.set_ylim(-5.5,5.5)
                    if seed==0:ax.set_ylabel(f'{modes} modes\n{method}')
                    if i==j==0:ax.set_title(f'Seed {seed}')
        fig.suptitle(f'{step//1000}k updates · first 3,000 fixed generated samples · × = component means')
        fig.tight_layout();fig.savefig(output/f'samples_{step//1000}k.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,3,figsize=(15,10))
        for i,modes in enumerate([9,25,49]):
            for j,method in enumerate(METHODS):
                ax=axes[i,j];mass=np.array([samples[(modes,method,s,step)][1] for s in range(5)])
                ax.bar(range(modes),mass.mean(0)*modes,color=COLORS[j]);ax.errorbar(range(modes),mass.mean(0)*modes,yerr=mass.std(0,ddof=1)*modes,fmt='none',ecolor='black',lw=.7)
                ax.axhline(1,color='black',ls='--');ax.set_title(f'{modes} modes · {method}');ax.set_xlabel('Mode index');ax.set_ylabel('Accepted mass / target mass')
        fig.suptitle(f'{step//1000}k updates · mean ± sample SD across five seeds');fig.tight_layout();fig.savefig(output/f'accepted_mass_{step//1000}k.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=True)
    for i,modes in enumerate([9,25,49]):
        config=configs[modes];ctr=centers(config).numpy();rng=np.random.default_rng(59100+modes)
        real=ctr[rng.integers(modes,size=3000)]+rng.normal(0,.1,(3000,2))
        points_list=[real]+[samples[(modes,m,0,50000)][0][:3000] for m in METHODS]
        for j,(points,title,color) in enumerate(zip(points_list,['True mixture','Vanilla, seed 0','RSGAN, seed 0','Paired, seed 0'],['#666666']+COLORS)):
            ax=axes[i,j];ax.scatter(points[:,0],points[:,1],s=.8,alpha=.4,color=color,rasterized=True)
            ax.scatter(ctr[:,0],ctr[:,1],marker='x',s=8,color='black');ax.set_aspect('equal');ax.set_xlim(-5.5,5.5);ax.set_ylim(-5.5,5.5)
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(f'{modes} components')
    fig.suptitle('Fixed spacing · true distribution versus 50k samples · shared coordinate scale')
    fig.tight_layout();fig.savefig(output/'reference_and_samples.png',dpi=160);plt.close(fig)
    for step in [10000,50000]:
        fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
        for i,modes in enumerate([9,25,49]):
            side=math.isqrt(modes)
            for j,method in enumerate(METHODS):
                ax=axes[i,j]
                ratio=np.mean([samples[(modes,method,seed,step)][1] for seed in range(5)],axis=0)*modes
                values=np.log2(np.clip(ratio,1/16,16)).reshape(side,side)
                im=ax.imshow(values,origin='lower',cmap='RdBu_r',vmin=-4,vmax=4)
                ax.set_title(f'{side}×{side} · {method}')
                axis=(np.arange(side)-(side-1)/2)*1.5
                ax.set_xticks(range(side),[f'{x:g}' for x in axis],fontsize=8)
                ax.set_yticks(range(side),[f'{x:g}' for x in axis],fontsize=8)
                ax.set_xlabel('Mode center x');ax.set_ylabel('Mode center y')
        cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,ticks=[-4,-2,0,2,4])
        cb.ax.set_yticklabels(['≤1/16×','1/4×','1× target','4×','≥16×'])
        cb.set_label('Mean accepted mode mass / ideal mass (log scale)')
        fig.suptitle(f'{step//1000}k updates · mean across five seeds · blue = underrepresented, red = overrepresented')
        fig.savefig(output/f'mode_mass_maps_{step//1000}k.png',dpi=160);plt.close(fig)
    sensitivity=[]
    for width in [.025,.05,.1]:
        edges=np.linspace(-5.5,5.5,round(11/width)+1)
        for modes in [9,25,49]:
            target=target_grid(configs[modes],edges)
            for step in [10000,50000]:
                for method in METHODS:
                    values=[spatial_tv(samples[(modes,method,seed,step)][0],target,edges) for seed in range(5)]
                    sensitivity.append({'cell_width':width,'modes':modes,'step':step,'method':method,'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))})
    (output/'grid_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    write_report(output, summary, reference)


def write_report(output, summary, reference):
    lines=['# Fixed-spacing Gaussian-grid sweep', '',
        '3×3, 5×5 and 7×7 grids (9, 25, 49 equally weighted components), centered at the origin, spacing 1.5 on each axis, sigma 0.1 per axis. Larger grids expand outward; no coordinate normalization. Adjacent centers remain 15 sigma apart and 3-sigma acceptance disks do not overlap.', '',
        '45 new unconditional training trajectories: vanilla, RSGAN and paired × five seeds × three grid sizes. Same original MLPs, latent dimension 16, widths 128, batch 256, Adam 0.0002 with betas (0.5,0.999), one D and one G update per step. Train to 50k; retain fixed 10,000-point evaluation draws at 10k and 50k. Coarse metrics every 1k. No method-specific tuning. The trainer is a frozen fork of the original with only the center layout changed (standalone ring CLI removed). Four local CPU workers, one torch thread each.', '',
        '## Findings', '',
        'At 50k, paired beats both baselines on mode TV and fine-grid TV in all five matched seeds for 3×3 and 5×5. On 3×3, paired covers all nine modes under both thresholds in every seed; vanilla and RSGAN each cover four at 50k and concentrate on the corners. On 5×5, paired covers 25/25 at the relative threshold in every seed and 25/25 at the original 1% threshold in four seeds (22/25 in seed 4). Baseline original-threshold coverage averages 7.2 and 7.8 modes.', '',
        'This advantage is budget-dependent: at 10k paired is behind the baselines on mean mode TV and fine-grid TV for 5×5, then overtakes them by 50k. On 7×7 paired remains behind at 50k: mean mode TV 0.710 versus 0.667 vanilla and 0.668 RSGAN; valid fraction 29.1% versus 37.2% and 38.8%. Its relative coverage averages 48.4/49, but original-threshold coverage only 6.8/49: it reaches many neighborhoods weakly while generating most points between modes. This is not successful recovery of the 49-component density.', '',
        'The sample panels show broad interior coverage for paired on the largest grid, while baselines emphasize edges and selected modes. At 50k all three have poor density fit there. Fine-grid mean rankings at 50k are unchanged at cell widths 0.025, 0.05 and 0.1. All models remain far from the real-sample density reference, including where paired wins mode balance.', '',
        'Paired mode TV on 7×7 is still improving through the endpoint (0.856, 0.802, 0.780, 0.756, 0.710 at 10k increments); no plateau or eventual winner is established. No training beyond the agreed 50k budget, architecture increase, normalization, or method-specific tuning was performed.', '',
        '## Metrics and controls', '',
        'Mode TV = ½(Σ|accepted mass per nearest mode − 1/K| + invalid fraction). Acceptance radius is 0.3. Invalid target mass is zero; true Gaussian tails put about 1.1% outside the acceptance disks, so the real distribution itself does not score exactly zero. All generated samples remain in the denominator.', '',
        'Report both original coverage (at least 1% of ALL generated points) and target-relative coverage (at least 8% of the ideal 1/K mass). The latter preserves the original eight-mode cutoff as a fraction of target mass. The 1% cutoff is increasingly strict: it requires 9%, 25%, 49% of target mass on these grids. Mode TV, per-mode mass and sample plots remain more informative than thresholded coverage.', '',
        'Fine-grid TV additionally measures density placement and shape: exact Gaussian-mixture probabilities in 0.05×0.05 cells over [-5.5,5.5]² plus an outside category versus generated sample counts. Every size uses the same unscaled spatial grid. A reference from 200 independent true-mixture draws of 10,000 points accounts for its nonzero finite-sample score; the interval is real-sampling variability, not uncertainty over GAN training. Cell widths 0.025 and 0.1 are retained as sensitivity checks.', '',
        'Although local separation is held fixed, expanding the grid increases coordinate range, distribution variance, and modes per fixed-size batch/network. These are the intended fixed-spacing scaling conditions; results do not isolate mode count from global extent or capacity. The 10k and 50k endpoints share a trajectory, and methods share seed-specific initialization/noise streams.', '',
        '## Results', '', 'Mean ± sample SD over five seeds. Relative coverage uses 8% of target mass.', '',
        '| Modes | Updates | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid samples | Coverage ≥1% | Relative coverage | Full relative-coverage seeds |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
        lines.append(f"| {r['modes']} | {r['step']} | {r['method']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['valid_fraction']['mean']:.1%} | {r['coverage_1pct']['mean']:.1f}/{r['modes']} | {r['coverage_relative']['mean']:.1f}/{r['modes']} | {r['full_relative_coverage_seeds']}/5 |")
    lines+=['','![Mode-count comparison](mode_count_comparison.png)','','![Training trajectories](training_curves.png)','','## Real-sample reference','','| Modes | Fine-grid TV mean | Central 95% interval |','|---:|---:|---|']
    for m,r in reference.items():
        lo,hi=r['real_spatial_tv_95_interval']
        lines.append(f"| {m} | {r['real_spatial_tv_mean']:.4f} | {lo:.4f}–{hi:.4f} |")
    lines+=['','## Samples and per-mode mass','','![True mixture and seed-0 samples](reference_and_samples.png)','','- [All-seed samples at 10k](samples_10k.png)','- [All-seed samples at 50k](samples_50k.png)','- [Accepted mass at 10k](accepted_mass_10k.png)','- [Accepted mass at 50k](accepted_mass_50k.png)','- [Spatial mode-mass maps at 10k](mode_mass_maps_10k.png)','- [Spatial mode-mass maps at 50k](mode_mass_maps_50k.png)','','Mode indices run left-to-right along each row, from the lowest y to highest y. The sample panels use the same coordinate limits for all grid sizes. Seed 0 is displayed for illustration; the all-seed panels and mean±SD use every seed.','','## Reproduce and verification','','Run `uv run python -m paired_discriminator.grid_mode_sweep` with fresh output directories. Configs are `configs/grid3-50k.json`, `grid5-50k.json`, `grid7-50k.json`. The launcher refuses overwrites. Build this report with `uv run python -m paired_discriminator.grid_mode_sweep_report results/grid-mode-count-v1`. All 90 endpoints are recalculated from saved sample arrays and checked against training logs; source snapshots and sample hashes are verified. Full metrics, thresholds, source snapshots, grid-resolution sensitivity, and checkpoints are retained. No prior results are overwritten.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    import sys
    build(sys.argv[1])
