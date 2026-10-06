"""Audit and report exact 7x7 continuations to 100k and 150k."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_experiment import ROOT, centers, coverage_metrics, models
from .grid_mode_sweep_report import target_grid, spatial_tv

METHODS=['vanilla','rsgan','paired']
COLORS=['#4477AA','#EEAA33','#228855']
STEPS=[50000,100000,150000]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=ROOT/'results/grid7-50k-v1';new=ROOT/'results/grid7-150k-v1'
    config=json.loads((new/'config.json').read_text())
    assert {**config,'steps':50000}==json.loads((old/'config.json').read_text())
    for src in [old,new]:
        provenance=json.loads((src/'provenance.json').read_text())
        for name,h in provenance['source_sha256'].items():assert digest(src/'source'/name)==h
    target=target_grid(config);rows=[];histories=[];samples={};verified={}
    for method in METHODS:
        for seed in range(5):
            name=f'{method}_seed{seed}';before=old/name;after=new/name
            resume=json.loads((after/'resume.json').read_text())
            assert resume['step']==50000 and resume['checkpoint_sha256']==digest(before/'checkpoint.pt')
            with (before/'metrics.csv').open() as f:initial=list(csv.DictReader(f))
            with (after/'metrics.csv').open() as f:extension=list(csv.DictReader(f))
            assert len(initial)==51 and len(extension)==101
            assert int(extension[0]['step'])==50000 and int(extension[-1]['step'])==150000
            for k in ['coverage','mode_tv','valid_fraction']+[f'mode_{i}_mass' for i in range(49)]:
                assert float(initial[-1][k])==float(extension[0][k])
            allrows=initial+extension[1:]
            for r in allrows:
                mass=np.array([float(r[f'mode_{i}_mass']) for i in range(49)])
                histories.append({'method':method,'seed':seed,'step':int(r['step']),
                    'mode_tv':float(r['mode_tv']),'valid_fraction':float(r['valid_fraction']),
                    'coverage_1pct':int(r['coverage']),'coverage_relative':int((mass>=.08/49).sum())})
            for step in STEPS:
                path=before/'final_samples.npy' if step==50000 else after/('samples_step100000.npy' if step==100000 else 'final_samples.npy')
                checkpoint=before/'checkpoint.pt' if step==50000 else after/('checkpoint_100000.pt' if step==100000 else 'checkpoint.pt')
                points=np.load(path);assert points.shape==(10000,2) and np.isfinite(points).all()
                c=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert c['step']==step and c['method']==method and c['seed']==seed
                assert {**c['config'],'steps':150000}==config
                g,_=models(config,method,seed);g.load_state_dict(c['generator']);g.eval()
                z=torch.randn(10000,config['latent_dim'],generator=torch.Generator().manual_seed(seed+70000))
                with torch.inference_mode():np.testing.assert_array_equal(g(z).numpy(),points)
                metrics=coverage_metrics(torch.from_numpy(points),config)
                stored=next(r for r in allrows if int(r['step'])==step)
                for k,v in metrics.items():assert abs(v-float(stored[k]))<1e-10
                mass=np.array([metrics[f'mode_{i}_mass'] for i in range(49)])
                rows.append({'method':method,'seed':seed,'step':step,'mode_tv':metrics['mode_tv'],
                    'spatial_tv':spatial_tv(points,target),'valid_fraction':metrics['valid_fraction'],
                    'coverage_1pct':metrics['coverage'],'coverage_relative':int((mass>=.08/49).sum())})
                samples[(method,seed,step)]=(points,mass)
                verified[f'{name}_{step}']={'samples_sha256':digest(path),'checkpoint_sha256':digest(checkpoint),'sample_replay_bit_exact':True}
    summary=[]
    for step in STEPS:
        for method in METHODS:
            rs=[r for r in rows if r['step']==step and r['method']==method]
            stats={'step':step,'method':method}
            for metric in ['mode_tv','spatial_tv','valid_fraction','coverage_1pct','coverage_relative']:
                values=[r[metric] for r in rs];stats[metric]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))}
            for metric in ['coverage_1pct','coverage_relative']:stats['full_'+metric+'_seeds']=sum(r[metric]==49 for r in rs)
            summary.append(stats)
    for name,data in [('endpoints.csv',rows),('learning_curves.csv',histories)]:
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'verified_endpoints':45,'new_endpoints':30,'resume_boundaries_verified':15,'records':verified},indent=2)+'\n')
    ref=json.loads((ROOT/'results/grid-mode-count-v1/reference_calibration.json').read_text())['49']
    (output/'reference_calibration.json').write_text(json.dumps(ref,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,(metric,title) in zip(axes,[('mode_tv','Mode TV ↓'),('valid_fraction','Valid sample fraction ↑'),('coverage_relative','Covered modes (≥8% of target mass) ↑')]):
        for method,color in zip(METHODS,COLORS):
            a=[]
            for seed in range(5):
                rr=[r for r in histories if r['method']==method and r['seed']==seed]
                x=np.array([r['step'] for r in rr])/1000;y=np.array([r[metric] for r in rr]);a.append(y)
                ax.plot(x,y,color=color,alpha=.15,lw=.6)
            a=np.array(a);mean=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(x,mean,color=color,label=method);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1)
        ax.axvline(50,color='black',ls='--',lw=1);ax.set_xlabel('Total training updates (thousands)');ax.set_title(title);ax.set_ylim(0,50 if metric=='coverage_relative' else 1);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('7×7 grid · exact continuation from 50k · five seeds · thin seed curves, mean ±SD')
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,(metric,title) in zip(axes,[('mode_tv','Mode TV ↓'),('spatial_tv','Fine-grid density TV ↓'),('coverage_1pct','Covered modes (≥1% of all samples) ↑')]):
        for method,color in zip(METHODS,COLORS):
            rr=[r for r in summary if r['method']==method]
            ax.errorbar([r['step']/1000 for r in rr],[r[metric]['mean'] for r in rr],yerr=[r[metric]['sd'] for r in rr],color=color,marker='o',capsize=3,label=method)
        if metric=='spatial_tv':ax.axhline(ref['real_spatial_tv_mean'],color='black',ls='--',label='real-sample reference')
        ax.set_xlabel('Total updates (thousands)');ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0].legend();axes[1].legend(fontsize=8);fig.suptitle('7×7 grid · 10,000 fixed samples per checkpoint · mean ±SD across five seeds')
    fig.tight_layout();fig.savefig(output/'endpoint_comparison.png',dpi=160);plt.close(fig)
    ctr=centers(config).numpy()
    for step in STEPS:
        fig,axes=plt.subplots(3,5,figsize=(13,8),sharex=True,sharey=True)
        for i,method in enumerate(METHODS):
            for seed in range(5):
                ax=axes[i,seed];points=samples[(method,seed,step)][0]
                ax.scatter(points[:3000,0],points[:3000,1],s=.8,alpha=.35,color=COLORS[i],rasterized=True);ax.scatter(ctr[:,0],ctr[:,1],s=8,marker='x',color='black')
                ax.set_aspect('equal');ax.set_xlim(-5.5,5.5);ax.set_ylim(-5.5,5.5)
                if seed==0:ax.set_ylabel(method)
                if i==0:ax.set_title(f'Seed {seed}')
        fig.suptitle(f'7×7 grid · {step//1000}k · first 3,000 fixed samples · × = component means')
        fig.tight_layout();fig.savefig(output/f'samples_{step//1000}k.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
    for i,step in enumerate(STEPS):
        for j,method in enumerate(METHODS):
            ratio=np.mean([samples[(method,seed,step)][1] for seed in range(5)],axis=0)*49
            ax=axes[i,j];im=ax.imshow(np.log2(np.clip(ratio,1/16,16)).reshape(7,7),origin='lower',cmap='RdBu_r',vmin=-4,vmax=4)
            ax.set_title(f'{step//1000}k · {method}');axis=(np.arange(7)-3)*1.5
            ax.set_xticks(range(7),axis,fontsize=8);ax.set_yticks(range(7),axis,fontsize=8)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,ticks=[-4,-2,0,2,4]);cb.ax.set_yticklabels(['≤1/16×','1/4×','1× target','4×','≥16×'])
    cb.set_label('Mean accepted mode mass / target (log scale)');fig.suptitle('7×7 grid · five-seed mean · blue = underrepresented, red = overrepresented')
    fig.savefig(output/'mode_mass_maps.png',dpi=160);plt.close(fig)
    comparisons=[]
    for step in STEPS:
        for baseline in METHODS[:2]:
            d={'step':step,'baseline':baseline}
            for metric in ['mode_tv','spatial_tv']:
                lookup={(r['method'],r['seed']):r[metric] for r in rows if r['step']==step}
                d[metric+'_paired_wins']=sum(lookup[('paired',s)]<lookup[(baseline,s)] for s in range(5))
            comparisons.append(d)
    (output/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    write_report(output,summary,ref)


def write_report(root,summary,ref):
    lines=['# 7×7 grid: exact continuation to 150k', '',
        'Continue all fifteen original 7×7 trajectories (vanilla, RSGAN, paired; seeds 0–4) from 50k for another 100k updates. Evaluate total updates 100k and 150k; retain 50k as the baseline. No restarts, architecture changes, rescaling, or tuning.', '',
        'Spacing 1.5, Gaussian sigma 0.1, extent ±4.5, latent 16, G/D width 128, batch 256, Adam 0.0002 with betas (0.5,0.999), one D and G update per iteration. Restore both models, Adam states and all explicit RNG streams. Recreate the same fixed 10,000-point noise draw, verify its 50k samples exactly, then restore the saved RNG states before further training. An uninterrupted-versus-split test confirms bit-exact weights, optimizer states, RNGs and final samples for all three methods.', '',
        'A new directory preserves the original 50k artifacts. Extension training_seconds/wall_seconds measure only this continuation. Coarse metrics remain evaluated every 1k; full sample arrays and full checkpoints are saved at 100k and 150k. All 45 reported endpoint sample arrays are regenerated bit-for-bit from their checkpoints, and metrics are recomputed from samples. Source snapshots and all 15 resume checkpoint hashes are checked.', '',
        '## Findings', '',
        'The ranking reverses with more training. Paired mean mode TV decreases from 0.710 at 50k to 0.352 at 100k and 0.231 at 150k, versus vanilla 0.667 → 0.564 → 0.520 and RSGAN 0.668 → 0.578 → 0.519. Paired beats both baselines on mode TV in all five matched seeds at both new endpoints. At 150k it also wins fine-grid density TV in all five matched seeds; at 100k it wins 4/5 versus vanilla and 5/5 versus RSGAN.', '',
        'At 150k, original-threshold coverage averages 41.4/49 for paired versus 21.0/49 for each baseline. Relative coverage averages 47.8/49 versus 34.4 and 35.0. Valid fractions are 78.2% paired, 76.4% vanilla and 74.8% RSGAN. Thus the advantage is principally allocation across modes rather than a large difference in total accepted mass. The baselines retain severe interior deficits; paired places much more mass in the interior.', '',
        'Recovery is incomplete: only one paired seed covers all 49 modes at the original 1% cutoff at 150k. Paired fine-grid TV is 0.647, well above the real-sample reference 0.264; the plots show imperfect Gaussian spread and bridges. Its mean fine-grid TV changes only modestly from 0.658 at 100k, while coarse mode TV continues improving. These untuned trajectories establish a late advantage in this setup, not a universal convergence rate or eventual equilibrium.', '',
        '## Metrics', '',
        'Mode TV = ½(Σ|accepted mass − 1/49| + invalid fraction), with acceptance radius 0.3 around the nearest center. Invalid target mass is zero; true Gaussian tails yield approximately 1.1% invalid samples. No renormalization of accepted samples. Report both ≥1% of all generated samples and ≥8% of target mode mass coverage; the latter cutoff is 0.08/49. At 49 modes the original 1% cutoff requires 49% of ideal mode mass, so the two measures answer different questions.', '',
        'Fine-grid TV compares empirical probability in fixed 0.05×0.05 cells on [-5.5,5.5]² plus an outside category to exact Gaussian-mixture cell probabilities. Its real-sample reference is reused from the original grid audit (200 independent draws, 10,000 samples each). It detects misplaced mass and incorrect Gaussian spread as well as mode imbalance.', '',
        '## Endpoints', '', 'Mean ± sample SD across five seeds. Endpoints share training trajectories and fixed evaluation noise; they are not independent runs.', '',
        '| Updates | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid samples | Coverage ≥1% | Relative coverage |', '|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
        lines.append(f"| {r['step']} | {r['method']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['valid_fraction']['mean']:.1%} | {r['coverage_1pct']['mean']:.1f}/49 | {r['coverage_relative']['mean']:.1f}/49 |")
    lines+=['',f"Real fine-grid TV reference: {ref['real_spatial_tv_mean']:.4f}, central 95% interval {ref['real_spatial_tv_95_interval']}. This describes real-sample variation, not uncertainty over training.",'','![Training curves](training_curves.png)','','![Endpoints](endpoint_comparison.png)','','![Mode-mass maps](mode_mass_maps.png)','','## Samples','','- [50k: all seeds](samples_50k.png)','- [100k: all seeds](samples_100k.png)','- [150k: all seeds](samples_150k.png)','','## Reproduce','','Run `uv run python -m paired_discriminator.grid_extension_run` with fresh output directories and the retained original checkpoints. Config: `configs/grid7-150k.json`. Build this report with `uv run python -m paired_discriminator.grid_extension_report results/grid7-extension-comparison-v1`. Checkpoints are ignored by Git but retained locally; source snapshots, sample arrays, metrics and graphics are retained. This is a fixed-architecture, fixed-batch, untuned comparison, not a general ranking of GAN objectives.','']
    (root/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    import sys
    build(sys.argv[1])
