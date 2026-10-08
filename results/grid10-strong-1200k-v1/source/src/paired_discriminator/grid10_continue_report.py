"""Verify and compare 100-mode grid trajectories with expanded density bounds."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_reference import ROOT,models,base_method,coverage_metrics,centers
from .grid_mode_sweep_report import target_grid,spatial_tv
from .grid_reference_report import write_csv
from .grid10_run import METHODS
EDGES=np.linspace(-7.75,7.75,311)
NAMES={'vanilla':'Vanilla','paired':'Paired','vanilla_deficit':'Vanilla + D deficit','paired_deficit':'Paired + D deficit'}


def metrics(points,config,target):
    a=coverage_metrics(torch.from_numpy(points),config)
    mass=np.array([a[f'mode_{i}_mass'] for i in range(config['modes'])])
    return {'mode_tv':a['mode_tv'],'spatial_tv':spatial_tv(points,target,EDGES),'valid_fraction':a['valid_fraction'],
            'coverage_1pct':a['coverage'],'coverage_half_target':int((mass>=.5/config['modes']).sum()),'coverage_relative':int((mass>=.08/config['modes']).sum())},mass


def build():
    torch.set_num_threads(1)
    root=ROOT/'results/grid10-300k-v1';output=ROOT/'results/grid10-300k-comparison-v1';output.mkdir(parents=True,exist_ok=True)
    c=json.loads((root/'config.json').read_text());target=target_grid(c,EDGES)
    assert target[-1]<1e-12
    provenance=json.loads((root/'provenance.json').read_text())
    for name,h in provenance['source_sha256'].items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==h
    rows=[];history=[];hashes={};replays=0;rng_checks=0;masses={}
    for method in METHODS:
        for seed in range(5):
            path=root/f'{method}_seed{seed}'
            resume=json.loads((path/'resume.json').read_text())
            assert resume['start_step']==150000
            source=Path(resume['source_checkpoint'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==resume['sha256']
            assert np.array_equal(np.load(path/'samples_step150000.npy'),np.load(source.parent/'samples_step150000.npy'))
            with (path/'metrics.csv').open() as f:rr=list(csv.DictReader(f))
            assert [int(r['step']) for r in rr]==list(range(150000,300001,50000))
            for r in rr:
                mass=np.array([float(r[f'mode_{i}_mass']) for i in range(100)])
                history.append(dict(method=method,seed=seed,step=int(r['step']),mode_tv=float(r['mode_tv']),valid_fraction=float(r['valid_fraction']),coverage_1pct=int(r['coverage']),coverage_half_target=int((mass>=.005).sum()),coverage_relative=int((mass>=.0008).sum())))
            for step in (150000,200000,250000,300000):
                file=path/f'samples_step{step}.npy';points=np.load(file)
                assert points.shape==(10000,2) and np.isfinite(points).all()
                metric,mass=metrics(points,c,target);logged=next(r for r in rr if int(r['step'])==step)
                assert abs(metric['mode_tv']-float(logged['mode_tv']))<1e-10
                assert abs(metric['valid_fraction']-float(logged['valid_fraction']))<1e-10
                saved=torch.load(path/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
                assert saved['config']==c and saved['method']==method and saved['seed']==seed and saved['step']==step
                g,_=models(c,method,seed);g.load_state_dict(saved['generator'])
                z=torch.randn(10000,c['latent_dim'],generator=torch.Generator().manual_seed(seed+70000))
                with torch.no_grad():assert np.array_equal(g(z).numpy(),points)
                for optimizer in ('optimizer_g','optimizer_d'):
                    assert all(int(s['step'])==step for s in saved[optimizer]['state'].values())
                replays+=1
                if method.endswith('_deficit'):
                    prior=torch.load(root/f'{base_method(method)}_seed{seed}'/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
                    for key in ('real_g','noise_d','noise_g','slots','eval'):assert torch.equal(saved['rng_states'][key],prior['rng_states'][key])
                    rng_checks+=1
                rows.append(dict(method=method,seed=seed,step=step,**metric));masses[method,seed,step]=mass
                hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    write_csv(output/'endpoints.csv',rows);write_csv(output/'learning_curves.csv',history)
    rng=np.random.default_rng(48292);means=centers(c).numpy();cal=[]
    for _ in range(200):
        points=(means[rng.integers(100,size=10000)]+rng.normal(0,.1,size=(10000,2))).astype(np.float32)
        cal.append(metrics(points,c,target)[0])
    reference={k:{'mean':float(np.mean([r[k] for r in cal])),'interval95':np.quantile([r[k] for r in cal],[.025,.975]).tolist()} for k in cal[0]}
    (output/'reference_calibration.json').write_text(json.dumps({'draws':200,'samples_per_draw':10000,'seed':48292,'metrics':reference,'edges':[-7.75,7.75,.05]},indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'checkpoint_replays':replays,'G_noise_RNG_checks':rng_checks,'samples_sha256':hashes},indent=2)+'\n')
    summary=[]
    for step in (150000,200000,250000,300000):
        for method in METHODS:
            rr=[r for r in rows if r['method']==method and r['step']==step]
            summary.append({'method':method,'step':step,**{key:{'mean':float(np.mean([r[key] for r in rr])),'sd':float(np.std([r[key] for r in rr],ddof=1))} for key in cal[0]},'full_half_target_seeds':sum(r['coverage_half_target']==100 for r in rr),'full_relative_seeds':sum(r['coverage_relative']==100 for r in rr)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,key,title in zip(axes.flat,('mode_tv','valid_fraction','coverage_half_target','coverage_relative'),('Mode TV ↓','Valid mass ↑','Coverage ≥50% of target mass ↑','Coverage ≥8% of target mass ↑')):
        for method in METHODS:
            rr=[r for r in history if r['method']==method];steps=sorted({r['step'] for r in rr});m=[];s=[]
            for step in steps:
                v=[r[key] for r in rr if r['step']==step];m.append(np.mean(v));s.append(np.std(v,ddof=1))
            x=np.array(steps)/1000;m=np.array(m);s=np.array(s);ax.plot(x,m,label=NAMES[method]);ax.fill_between(x,m-s,m+s,alpha=.12)
        ax.set_title(title);ax.set_xlabel('G/D updates (thousands)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('10×10 fixed-spacing grid • 5 seeds • mean ± SD');fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,10))
    for ax,method in zip(axes.flat,METHODS):
        matrix=np.mean([masses[method,s,300000] for s in range(5)],axis=0).reshape(10,10)*100
        ax.imshow(matrix,origin='lower',vmin=0,vmax=1.5,cmap='YlGnBu');ax.set_title(NAMES[method]);ax.set_xticks([]);ax.set_yticks([])
        for i in range(10):
            for j in range(10):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=6)
    fig.suptitle('300k: accepted mass / target mass • mean of five seeds • target = 1');fig.tight_layout();fig.savefig(output/'mode_mass_maps.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for ax,method in zip(axes.flat,METHODS):
        points=np.load(root/f'{method}_seed0/samples_step300000.npy');ax.scatter(points[:,0],points[:,1],s=1,alpha=.25);ax.scatter(means[:,0],means[:,1],s=8,c='red',marker='+');ax.set_xlim(-7.75,7.75);ax.set_ylim(-7.75,7.75);ax.set_aspect('equal');ax.set_title(NAMES[method])
    fig.suptitle('300k fixed samples • seed 0, not selected for performance');fig.tight_layout();fig.savefig(output/'samples_seed0.png',dpi=170);plt.close(fig)
    lines=['# 10×10 grid: D-only deficit sampling','','100 Gaussian modes; spacing 1.5, sigma .1, centers span ±6.75. Five seeds through 300k. Same networks, losses, batches and deficit rule as prior grids.','','Coverage ≥1% is retained in CSVs but now equals the entire target mass per mode. The tables emphasize ≥50% and ≥8% of target mass; the latter is the existing lenient relative criterion.','','Density TV uses 0.05 cells over [-7.75,7.75]² plus outside mass, expanded to encompass all 100 modes. Do not directly compare density TV across mode counts without the finite-sample real reference.','',f"True-mixture 10k-sample reference: mean mode TV {reference['mode_tv']['mean']:.4f}, fine-density TV {reference['spatial_tv']['mean']:.4f}.",'','![Curves](learning_curves.png)']
    for step in (150000,200000,250000,300000):
        lines += ['',f'## {step:,} updates','','Mean ± sample SD across five seeds.','','| Method | Mode TV ↓ | Fine TV ↓ | Valid mass ↑ | Coverage ≥50% target | Coverage ≥8% target | Full ≥50% seeds |','|---|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in summary if r['step']==step]:
            def fmt(k):return f"{r[k]['mean']:.4f} ± {r[k]['sd']:.4f}"
            lines.append(f"| {NAMES[r['method']]} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {fmt('valid_fraction')} | {r['coverage_half_target']['mean']:.1f}/100 | {r['coverage_relative']['mean']:.1f}/100 | {r['full_half_target_seeds']}/5 |")
    lines+=['','![Mode mass](mode_mass_maps.png)','','Heatmaps average seeds; they do not imply every seed covers every mode.','','![Samples](samples_seed0.png)','',f'Verification: {replays} exact endpoint replays, {rng_checks} unchanged G/noise RNG checks. Source hashes and optimizer counts verified.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',output,flush=True)
if __name__=='__main__':build()
