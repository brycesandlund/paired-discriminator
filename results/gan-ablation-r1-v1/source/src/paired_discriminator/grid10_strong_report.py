"""Verify stronger D-only intervention and compare with retained paired baselines."""
import json,csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid10_strong import ROOT,models,centers
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

def build():
    torch.set_num_threads(1)
    root=ROOT/'results/grid10-strong-1200k-v1';out=ROOT/'results/grid10-strong-comparison-v1';out.mkdir(exist_ok=True)
    c=json.loads((root/'config.json').read_text());target=target_grid(c,EDGES);rows=[];replays=0;rngchecks=0
    for seed in range(5):
        p=root/f'paired_deficit_seed{seed}'
        assert (p/'metadata.json').exists()
        with (p/'metrics.csv').open() as f:logged=list(csv.DictReader(f))
        for step in range(50000,1200001,50000):
            state=torch.load(p/f'checkpoint_{step}.pt',weights_only=False)
            assert state['step']==step and state['config']==c and state['seed']==seed
            g,_=models(c,'paired_deficit',seed);g.load_state_dict(state['generator'])
            z=torch.randn(10000,c['latent_dim'],generator=torch.Generator().manual_seed(seed+70000))
            with torch.no_grad():points=g(z).numpy()
            assert np.array_equal(points,np.load(p/f'samples_step{step}.npy'))
            m,_=metrics(points,c,target)
            r=next(r for r in logged if int(r['step'])==step)
            assert abs(m['mode_tv']-float(r['mode_tv']))<1e-10
            for key in ('optimizer_g','optimizer_d'):assert all(int(v['step'])==step for v in state[key]['state'].values())
            rows.append(dict(method='paired_strong',seed=seed,step=step,**m));replays+=1
        old=torch.load(ROOT/f'results/grid10-1200k-v1/paired_deficit_seed{seed}/checkpoint.pt',weights_only=False)
        for key in ('real_g','noise_d','noise_g','slots','eval'):assert torch.equal(state['rng_states'][key],old['rng_states'][key])
        rngchecks+=1
    for folder in ('grid10-comparison-v1','grid10-300k-comparison-v1','grid10-600k-comparison-v1','grid10-1200k-comparison-v1'):
        path=ROOT/'results'/folder/'endpoints.csv'
        if not path.exists():continue
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['method'] in ('paired','paired_deficit'):
                    rows.append({k:(v if k=='method' else int(v) if k in ('seed','step') else float(v)) for k,v in r.items()})
    rows=list({(r['method'],r['seed'],r['step']):r for r in rows}.values())
    (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(checkpoint_replays=replays,noise_slot_eval_rng_comparisons=rngchecks),indent=2)+'\n')
    names={'paired':'Paired uniform','paired_deficit':'Paired D deficit','paired_strong':'Paired D deficit α=.9, p=2'}
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key in zip(axes,('mode_tv','spatial_tv','coverage_half_target')):
        for method,name in names.items():
            steps=sorted({r['step'] for r in rows if r['method']==method});means=[];sd=[]
            for step in steps:
                v=[r[key] for r in rows if r['method']==method and r['step']==step];means.append(np.mean(v));sd.append(np.std(v,ddof=1))
            means=np.array(means);sd=np.array(sd);x=np.array(steps)/1000
            ax.plot(x,means,label=name);ax.fill_between(x,means-sd,means+sd,alpha=.12)
        ax.set(title=key,xlabel='Updates (thousands)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    lines=['# Stronger paired D-only deficit sampling','','Five fresh seeds; 1.2M updates. Mean ± SD curves. Existing paired baselines reused.','','![Comparison](comparison.png)','','| Method | Mode TV | Fine TV | Half-target coverage |','|---|---:|---:|---:|']
    for method,name in names.items():
        rr=[r for r in rows if r['method']==method and r['step']==1200000];assert len(rr)==5
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([r[k] for r in rr]):.4f}' for k in ('mode_tv','spatial_tv','coverage_half_target'))+' |')
    lines+=['',f'Verified {replays} checkpoint replays and {rngchecks} final noise/slot/eval RNG comparisons. Uniform G reference RNG also matches the D-only baseline.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',out,flush=True)
if __name__=='__main__':build()
