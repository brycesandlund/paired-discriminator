"""Verified G-only deficit comparison against frozen previous grid experiments."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_reference import ROOT,models,coverage_metrics
from .grid_reference_report import read_csv,write_csv,budgets
from .grid_dual_slot_report import run_path
from .grid_mode_sweep_report import target_grid,spatial_tv

DISPLAY=['vanilla','paired','paired_deficit','paired_g_deficit']
NAMES={'vanilla':'Vanilla','paired':'Paired uniform','paired_deficit':'Paired D deficit','paired_g_deficit':'Paired G deficit'}

def build(output):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=ROOT/'results/grid-reference-comparison-v1'
    rows=read_csv(old/'endpoints.csv');history=read_csv(old/'learning_curves.csv')
    # Recompute previous coarse and density metrics; no historical results are overwritten.
    hashes={};replays=0;rng_checks=0
    for side in (3,5,7):
        root=ROOT/f'results/grid{side}-reference-g-v1'
        config=json.loads((root/'config.json').read_text());target=target_grid(config)
        for r in [r for r in rows if r['modes']==side*side]:
            file=ROOT/r['source'];points=np.load(file)
            actual=coverage_metrics(torch.from_numpy(points),config)
            assert abs(actual['mode_tv']-r['mode_tv'])<1e-10
            assert abs(spatial_tv(points,target)-r['spatial_tv'])<1e-10
            hashes[r['source']]=hashlib.sha256(file.read_bytes()).hexdigest()
        provenance=json.loads((root/'provenance.json').read_text())
        for name,digest in provenance['source_sha256'].items():
            assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest
        for seed in range(5):
            path=root/f'paired_deficit_seed{seed}'
            rr=read_csv(path/'metrics.csv')
            assert [int(r['step']) for r in rr]==list(range(0,config['steps']+1,1000))
            for r in rr:
                mass=np.array([float(r[f'mode_{i}_mass']) for i in range(side*side)])
                history.append(dict(modes=side*side,method='paired_g_deficit',seed=seed,step=int(r['step']),mode_tv=float(r['mode_tv']),valid_fraction=float(r['valid_fraction']),coverage_1pct=int(r['coverage']),coverage_relative=int((mass>=.08/(side*side)).sum())))
            for step in budgets(side):
                file=path/f'samples_step{step}.npy';points=np.load(file)
                assert points.shape==(10000,2) and np.isfinite(points).all()
                a=coverage_metrics(torch.from_numpy(points),config)
                logged=next(r for r in rr if int(r['step'])==step)
                for k,v in a.items():assert abs(v-float(logged[k]))<1e-10
                saved=torch.load(path/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
                assert saved['config']==config and saved['step']==step and saved['seed']==seed
                g,_=models(config,'paired',seed);g.load_state_dict(saved['generator'])
                z=torch.randn(10000,config['latent_dim'],generator=torch.Generator().manual_seed(seed+70000))
                with torch.no_grad():assert np.array_equal(g(z).numpy(),points)
                for optimizer in ('optimizer_d','optimizer_g'):
                    assert all(int(s['step'])==step for s in saved[optimizer]['state'].values())
                replays+=1
                if step>=50000:
                    base=run_path(side,'paired',step)/f'paired_seed{seed}'
                    prior=torch.load(base/('checkpoint_100000.pt' if step==100000 else 'checkpoint.pt'),map_location='cpu',weights_only=False)
                    assert prior['step']==step
                    for key in ('real_d','noise_d','noise_g','slots','eval'):
                        assert torch.equal(saved['rng_states'][key],prior['rng_states'][key])
                    rng_checks+=1
                mass=np.array([a[f'mode_{i}_mass'] for i in range(side*side)])
                source=str(file.relative_to(ROOT));hashes[source]=hashlib.sha256(file.read_bytes()).hexdigest()
                rows.append(dict(modes=side*side,method='paired_g_deficit',seed=seed,step=step,mode_tv=a['mode_tv'],spatial_tv=spatial_tv(points,target),valid_fraction=a['valid_fraction'],coverage_1pct=a['coverage'],coverage_relative=int((mass>=.08/(side*side)).sum()),source=source))
    write_csv(output/'endpoints.csv',rows);write_csv(output/'learning_curves.csv',history)
    (output/'verification.json').write_text(json.dumps({'checkpoint_replays':replays,'unchanged_D_and_noise_RNG_checks':rng_checks,'samples_sha256':hashes},indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,side in zip(axes,(3,5,7)):
        for method in DISPLAY:
            rr=[r for r in history if r['modes']==side*side and r['method']==method]
            steps=sorted({r['step'] for r in rr});means=[];stds=[]
            for step in steps:
                vals=[float(r['mode_tv']) for r in rr if r['step']==step];means.append(np.mean(vals));stds.append(np.std(vals,ddof=1))
            x=np.array(steps)/1000;means=np.array(means);stds=np.array(stds)
            ax.plot(x,means,label=NAMES[method]);ax.fill_between(x,means-stds,means+stds,alpha=.12)
        ax.set_title(f'{side}×{side} grid');ax.set_xlabel('G/D updates (thousands)');ax.set_ylabel('Mode TV ↓');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Reference deficit sampling: phase comparison • five seeds, mean ± SD')
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,10))
    for row,side in enumerate((3,5,7)):
        step=150000 if side==7 else 50000
        for col,method in enumerate(DISPLAY[1:]):
            rr=[r for r in rows if r['modes']==side*side and r['method']==method and r['step']==step]
            masses=[]
            for r in rr:
                a=coverage_metrics(torch.from_numpy(np.load(ROOT/r['source'])),json.loads((ROOT/f'results/grid{side}-reference-g-v1/config.json').read_text()))
                masses.append([a[f'mode_{i}_mass'] for i in range(side*side)])
            matrix=np.mean(masses,axis=0).reshape(side,side)*side*side
            ax=axes[row,col];im=ax.imshow(matrix,origin='lower',vmin=0,vmax=1.5,cmap='YlGnBu');ax.set_title(f'{NAMES[method]} · {side}×{side} · {step//1000}k');ax.set_xticks([]);ax.set_yticks([])
            for i in range(side):
                for j in range(side):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7)
    fig.suptitle('Accepted mass / target mass per mode • average of five seeds • target = 1')
    fig.tight_layout();fig.savefig(output/'mode_mass_maps.png',dpi=170);plt.close(fig)
    lines=['# G-only deficit references on grids','','D samples real points uniformly; only paired G references are deficit-weighted. Same EMA .99 and 50% uniform floor as D-only experiment. Five seeds; no new vanilla training.','', '![Learning curves](learning_curves.png)','','## Final endpoints','','Mean ± sample SD. Coverage uses ≥1% of all generated samples per mode.','','| Grid | Method | Mode TV ↓ | Fine density TV ↓ | Valid mass ↑ | Coverage ↑ | Full coverage seeds |','|---|---|---:|---:|---:|---:|---:|']
    for side in (3,5,7):
        step=150000 if side==7 else 50000
        for method in dict.fromkeys(r['method'] for r in rows):
            rr=[r for r in rows if r['modes']==side*side and r['method']==method and r['step']==step]
            if not rr:continue
            def fmt(key):
                v=[float(r[key]) for r in rr];return f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}'
            lines.append(f"| {side}×{side} | {NAMES.get(method,method)} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {fmt('valid_fraction')} | {np.mean([r['coverage_1pct'] for r in rr]):.1f}/{side*side} | {sum(r['coverage_1pct']==side*side for r in rr)}/5 |")
    lines+=['','![Mode masses](mode_mass_maps.png)','',f'Verification: {replays} exact checkpoint replays; {rng_checks} D/noise RNG comparisons; historical mode and fine-density metrics recomputed. All failures retained. Full endpoint and learning-curve data in CSV files.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',output,flush=True)

if __name__=='__main__':build(ROOT/'results/grid-reference-g-comparison-v1')
