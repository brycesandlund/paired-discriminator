"""Fresh-noise confirmation, source/replay checks, and full ablation report."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .flow_ablation import Field
from .grid10_flow import sample
from .grid10_flow_oracle import Oracle
from .grid_experiment import ROOT,centers
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

def build():
    torch.set_num_threads(1)
    out=ROOT/'results/flow-ablation-comparison-v1';out.mkdir(exist_ok=True)
    roots=[ROOT/'results'/n for n in ['flow-ablation-r1-v1','flow-ablation-r2-v1','flow-ablation-validation-v1']]
    c=json.loads((roots[-1]/'config.json').read_text());target=target_grid(c,EDGES);oracle=Oracle(c)
    for root in roots:
        h=json.loads((root/'provenance.json').read_text())['source_sha256']
        for name,value in h.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==value
    rows=[];replays=0;sample_arrays={};field_errors=[];metadata={}
    for seed in range(5):
        z=torch.randn(10000,2,generator=torch.Generator().manual_seed(902000+seed))
        nets=[('analytic',oracle,None)]
        for name in ['cosine','two_layers']:
            path=(roots[1] if seed==0 else roots[2])/f'{name}_seed{seed}'
            ck=torch.load(path/'checkpoint_50000.pt',weights_only=False)
            assert ck['step']==50000 and ck['seed']==seed and not ck['spec'].get('oracle',False)
            assert all(int(s['step'])==50000 for s in ck['optimizer']['state'].values())
            net=Field(ck['spec'],ck['scale']);net.load_state_dict(ck['model'])
            fixed=torch.randn(10000,2,generator=torch.Generator().manual_seed(70000+seed))
            assert np.array_equal(sample(net,fixed,128).numpy(),np.load(path/'samples_50000_raw.npy'));replays+=1
            metadata[name,seed]=json.loads((path/'metadata.json').read_text());nets.append((name,net,path))
        for name,net,path in nets:
            for steps in (128,256):
                tick=time.perf_counter()
                points=torch.cat([sample(net,b,steps) for b in z.split(1000)]).numpy() if name=='analytic' else sample(net,z,steps).numpy()
                seconds=time.perf_counter()-tick;metric,mass=metrics(points,c,target)
                rows.append(dict(name=name,seed=seed,solver_steps=steps,sampling_seconds=seconds,**metric))
                np.save(out/f'{name}_seed{seed}_solver{steps}.npy',points)
                sample_arrays[name,seed,steps]=points
        # Field error evaluated on fresh true interpolation marginals, never used for training.
        rng=torch.Generator().manual_seed(913000+seed);x0=torch.randn(5000,2,generator=rng)
        mu=centers(c);x1=mu[torch.randint(100,(5000,),generator=rng)]+.1*torch.randn(5000,2,generator=rng)
        with torch.no_grad():
            for tt in [.25,.5,.75,.9,.95,.99]:
                t=torch.full((len(x0),1),tt);xt=(1-t)*x0+t*x1;v=oracle(xt,t)
                for name,net,_ in nets[1:]:field_errors.append(dict(name=name,seed=seed,t=tt,rmse=float((net(xt,t)-v).square().mean().sqrt())))
    (out/'validation_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'field_errors.json').write_text(json.dumps(field_errors,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(final_checkpoint_replays=replays,source_hash_checks=True,optimizer_counts=True,independent_noise_seeds=[902000+i for i in range(5)],chosen_variant='raw',selection_seed=0,confirmation_seeds=[1,2,3,4]),indent=2)+'\n')
    # Full search record, no hidden failed or oracle-supervised attempts.
    trials=[]
    for root in roots:
        for p in sorted(root.glob('*/metrics.json')):
            spec=next(s for s in json.loads((root/'specs.json').read_text()) if p.parent.name.startswith(s['name']+'_seed'))
            for r in json.loads(p.read_text()):trials.append(dict(round=root.name,oracle=spec.get('oracle',False),**r))
    (out/'all_trials.json').write_text(json.dumps(trials,indent=2)+'\n')
    lines=['# Learned flow matching approaches the analytic sampler','','All trials retained. Sample-only training remains independent-coupling straight-line flow matching. Known mixture parameters are used only for oracle diagnostics and evaluation. Model choices selected on seed 0; seeds 1–4 are confirmation runs. Raw (not EMA) weights chosen before confirmation. Fresh evaluation noise is independent of model-selection noise.','', '## Independent-noise results at 50k updates','','Five-seed mean ± sample SD. Identical 10k noise draws per seed across models and solvers.','','| Model | Midpoint steps | Mode TV ↓ | Fine TV ↓ | Valid mass ↑ | Half-target coverage ↑ |','|---|---:|---:|---:|---:|---:|']
    names={'analytic':'Exact analytic field','two_layers':'Learned, 2 hidden layers','cosine':'Learned, 3 hidden layers'}
    for name in ['analytic','two_layers','cosine']:
        for steps in (128,256):
            rr=[r for r in rows if r['name']==name and r['solver_steps']==steps]
            vals=[f'{np.mean([r[k] for r in rr]):.4f} ± {np.std([r[k] for r in rr],ddof=1):.4f}' for k in ['mode_tv','spatial_tv','valid_fraction','coverage_half_target']]
            lines.append(f'| {names[name]} | {steps} | '+' | '.join(vals)+' |')
    lines += ['','## What changed','','Estimate scalar data standard deviation from 100k pilot samples. At each time provide x/s_t, where s_t²=(1-t)²+t²s_data², together with raw t and sinusoidal features of both. Ten generic frequencies from .5 to 16 cycles per input coordinate. Predict a residual added to Gaussian base velocity [t s_data²-(1-t)]x/s_t². This base uses only estimated data scale; it contains no mode centers, labels, spacing or component sigma.','','Two/three width128 SiLU hidden layers; Adam (.9,.999), lr .001 decayed by cosine to .0001 over 50k updates; batch256; uniform times and independent noise–real pairs. No oracle labels, OT coupling or late-time reweighting in confirmed models.','','| Model | Parameters | Mean training seconds | Real draws per run |','|---|---:|---:|---:|']
    for name in ['two_layers','cosine']:
        mm=[metadata[name,s] for s in range(5)];lines.append(f"| {names[name]} | {mm[0]['parameters']} | {np.mean([m['training_seconds'] for m in mm]):.2f} | 12.8M + shared 100k pilot |")
    lines += ['','Wall times vary with concurrent workloads; these are measurements, not rigorous throughput benchmarks. Generation uses 256/512 network evaluations at 128/256 midpoint steps. GAN generation is one pass.','','## Screening (seed 0 only)','','Round 1 uses 4k evaluation samples; round 2 uses 10k. Fine TV is sample-size sensitive: do not compare its values across rounds without calibration. Oracle-supervised trials are diagnostics, not learned baselines.','','| Round | Arm | Updates | Raw/EMA | Mode TV | Fine TV | Valid | Half-target |','|---|---|---:|---|---:|---:|---:|---:|']
    for r in trials:
        if r['round'] in [roots[0].name,roots[1].name] and r['step'] in [20000,50000]:
            label=r['name']+(' (ORACLE SUPERVISION)' if r['oracle'] else '')
            lines.append(f"| {1 if r['round']==roots[0].name else 2} | {label} | {r['step']} | {r['variant']} | {r['mode_tv']:.4f} | {r['spatial_tv']:.4f} | {r['valid_fraction']:.4f} | {r['coverage_half_target']} |")
    lines+=['','## Samples','','Seed 0 fixed in advance. New analytic/tuned models use identical fresh noise. Original flow is the retained 1.2M seed-0 sample for context; original latent draws differ.','','![Samples](samples.png)','','## Limits','','This establishes near-oracle distributional fit on this grid and these metrics, not equality of velocity fields everywhere. The architecture was tuned on this problem. Five seeds include the development seed; four confirmation seeds are identified above. Oracle teacher trials use privileged information and are excluded from confirmed learned results. Source hashes, final samples and optimizer counts verified.']
    fig,axes=plt.subplots(2,2,figsize=(11,11));mu=centers(c).numpy()
    orig=np.load(ROOT/'results/grid10-flow-1200k-v1/seed0/samples_1200000_solver128.npy')
    pts=[orig,sample_arrays['analytic',0,128],sample_arrays['two_layers',0,128],sample_arrays['cosine',0,128]]
    titles=['Original learned flow • 1.2M','Exact analytic field','Tuned 2-layer flow • 50k','Tuned 3-layer flow • 50k']
    for ax,p,title in zip(axes.flat,pts,titles):
        ax.scatter(p[:,0],p[:,1],s=2,alpha=.25);ax.scatter(mu[:,0],mu[:,1],s=7,c='red');ax.set(title=title,xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal')
    fig.suptitle('10×10 grid • 10,000 samples • seed 0 • midpoint128');fig.tight_layout();fig.savefig(out/'samples.png',dpi=170);plt.close(fig)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:20]),flush=True)
if __name__=='__main__':build()
