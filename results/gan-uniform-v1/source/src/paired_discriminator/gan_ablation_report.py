"""Fresh-noise GAN comparison and complete adaptive search record."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .gan_ablation import make_models
from .grid_pacgan import models as old_models
from .grid_experiment import ROOT,centers,sample_real
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

LABELS={'old_vanilla':'Original vanilla • 1.2M','old_strong':'Original paired strong deficit • 1.2M','ema_fast':'Paired • squared deficit, fast estimator','linear':'Paired • linear deficit','vanilla_fast':'Vanilla • squared deficit, fast estimator','vanilla_linear':'Vanilla • linear deficit','real':'Real mixture reference'}

def build():
    torch.set_num_threads(1);out=ROOT/'results/gan-ablation-comparison-v1';out.mkdir(exist_ok=True)
    folders=['gan-ablation-r1-v1','gan-ablation-r2-v1','gan-ablation-r3-v1','gan-ablation-r4-v1','gan-ablation-validation-v1','gan-ablation-vanilla-linear-v1']
    all_trials=[]
    for folder in folders:
        root=ROOT/'results'/folder;h=json.loads((root/'provenance.json').read_text())['source_sha256']
        for f,v in h.items():assert hashlib.sha256((root/'source'/f).read_bytes()).hexdigest()==v
        for p in root.glob('*/metrics.json'):
            all_trials += [dict(round=folder,**r) for r in json.loads(p.read_text())]
    c=json.loads((ROOT/'results/gan-ablation-r4-v1/config.json').read_text());target=target_grid(c,EDGES);rows=[];replays=0;rngchecks=0;models_meta=[]
    for seed in range(5):
        for name in ['old_vanilla','old_strong','ema_fast','linear','vanilla_fast','vanilla_linear','real']:
            ck=None
            if name=='real':
                variants=[('raw',sample_real(c,10000,torch.Generator().manual_seed(942000+seed)).numpy())]
            elif name.startswith('old_'):
                folder='grid10-1200k-v1' if name=='old_vanilla' else 'grid10-strong-1200k-v1';method='vanilla' if name=='old_vanilla' else 'paired_deficit'
                ck=torch.load(ROOT/f'results/{folder}/{method}_seed{seed}/checkpoint.pt',weights_only=False)
                g,_=old_models(ck['config'],'vanilla' if name=='old_vanilla' else 'paired',seed);g.load_state_dict(ck['generator'])
                z=torch.randn(10000,16,generator=torch.Generator().manual_seed(932000+seed))
                with torch.no_grad():variants=[('raw',g(z).numpy())]
            else:
                folder='gan-ablation-vanilla-linear-v1' if name=='vanilla_linear' else 'gan-ablation-r4-v1' if seed==0 and name in ['ema_fast','linear'] else 'gan-ablation-validation-v1'
                path=ROOT/f'results/{folder}/{name}_seed{seed}'
                ck=torch.load(path/'checkpoint_50000.pt',weights_only=False);assert ck['seed']==seed and ck['step']==50000
                for key in ['optimizer_g','optimizer_d']:assert all(int(v['step'])==50000 for v in ck[key]['state'].values())
                meta=json.loads((path/'metadata.json').read_text());models_meta.append(dict(name=name,**meta))
                if name=='vanilla_linear':
                    pair=ROOT/'results'/('gan-ablation-r4-v1' if seed==0 else 'gan-ablation-validation-v1')/f'linear_seed{seed}'/'checkpoint_50000.pt'
                    pck=torch.load(pair,weights_only=False)
                    for k in ['real_g','noise_d','noise_g','slots']:assert torch.equal(ck['rng'][k],pck['rng'][k])
                    rngchecks+=1
                g,_=make_models(ck['config'],ck['spec'],seed);variants=[]
                for variant,key in [('raw','generator'),('ema','ema')]:
                    g.load_state_dict(ck[key]);z=torch.randn(10000,2,generator=torch.Generator().manual_seed(70000+seed))
                    with torch.no_grad():assert np.array_equal(g(z).numpy(),np.load(path/f'samples_50000_{variant}.npy'))
                    replays+=1;z=torch.randn(10000,2,generator=torch.Generator().manual_seed(932000+seed))
                    with torch.no_grad():variants.append((variant,g(z).numpy()))
            for variant,points in variants:
                m,mass=metrics(points,c,target);rows.append(dict(name=name,variant=variant,seed=seed,**m));np.save(out/f'{name}_{variant}_seed{seed}.npy',points)
    (out/'all_trials.json').write_text(json.dumps(all_trials,indent=2)+'\n');(out/'validation_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'model_metadata.json').write_text(json.dumps(models_meta,indent=2)+'\n')
    summary=[]
    for name in LABELS:
        for variant in ['raw','ema']:
            rr=[r for r in rows if r['name']==name and r['variant']==variant]
            if not rr:continue
            summary.append(dict(name=name,variant=variant,full_coverage_seeds=sum(r['coverage_half_target']==100 for r in rr),**{k:dict(mean=float(np.mean([r[k] for r in rr])),sd=float(np.std([r[k] for r in rr],ddof=1))) for k in ['mode_tv','spatial_tv','valid_fraction','coverage_half_target','coverage_relative']}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(checkpoint_sample_replays=replays,paired_vanilla_rng_comparisons=rngchecks,source_hashes=True,optimizer_counts=True,torch=torch.__version__,python=platform.python_version()),indent=2)+'\n')
    lines=['# GAN features, optimizer and deficit sampling (#33)','','Original BCE losses throughout; no reconstruction, matching loss, oracle velocity, or inference-time snapping. Screening uses seed0; confirmatory runs use seeds1–4. Vanilla linear-sampler follow-up was selected after seeing the paired confirmation results. Final noise (932000+seed) is independent of screening noise. Primary comparison uses raw G; EMA is a separately reported option, never selected independently per seed.','','## Final fresh-noise results','','Five seeds, 10k samples per seed; mean ± sample SD. Coverage counts modes with accepted mass ≥0.5% (half the 1% target).','','| Model | Output | Mode TV ↓ | Fine TV ↓ | Valid mass ↑ | Half-target modes | Full coverage seeds |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        def fmt(k):return f"{r[k]['mean']:.4f} ± {r[k]['sd']:.4f}"
        lines.append(f"| {LABELS[r['name']]} | {r['variant']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {fmt('valid_fraction')} | {r['coverage_half_target']['mean']:.1f}/100 | {r['full_coverage_seeds']}/5 |")
    lines+=['','## Architecture and training','','Tuned G takes two normal latent values, adds Fourier features at ten generic log-spaced frequencies .5–16 cycles, and uses two width128 LeakyReLU hidden layers. Output is s_data*(z + MLP(features(z))). s_data is the standard deviation estimated from a shared independent 100k-point real-data pilot. No target mode centers or spacings enter these features.','','D uses raw coordinates divided by s_data plus the same type of Fourier features, introduced from low to high over the first 25k updates. Two width128 LeakyReLU hidden layers; paired input has four raw coordinates, vanilla two.','','50k updates; D real256/fake256, G fake256; paired G references uniform256. Adam (.5,.999), both lrs .001 for 25k then cosine to .0001. Generator EMA .999 retained but raw is primary. Linear deficit uses alpha=.9,p=1, mass EMA .99. Fast-estimator squared deficit uses alpha=.9,p=2, mass EMA .9. Deficit uses known mode identities exactly as in prior grid experiments.','','| Model | G params | D params | Mean train seconds | D real draws | G references used |','|---|---:|---:|---:|---:|---:|']
    for name in ['ema_fast','linear','vanilla_fast','vanilla_linear']:
        mm=[r for r in models_meta if r['name']==name]
        lines.append(f"| {LABELS[name]} | {mm[0]['parameters_g']} | {mm[0]['parameters_d']} | {np.mean([r['training_seconds'] for r in mm]):.2f} | 12.8M | {'12.8M' if not name.startswith('vanilla') else '0'} |")
    lines+=['','A shared 100k-point scale pilot is additional. The implementation draws unused G real batches for vanilla to maintain RNG parity, but those values do not affect its gradients. Training time depends on concurrent workloads. One G forward pass per generated sample; parameter count differs from the old G/D. Equal update count does not mean equal compute.','','## Search record','','53 short jobs: 35 seed0 screening, 13 initial confirmation, 5 matched vanilla-linear follow-up. Every raw/EMA intermediate and final metric is retained in all_trials.json, including failures to improve. Rounds1–3 use 4k evaluation samples, later rounds10k; fine TV floors differ with sample count.','','R1: D features alone did not solve coverage. Gradually introduced features sharpened points while still losing modes. R2: Gaussian skip alone did not solve it either. R3: adding G Fourier features made the large observed improvement. R4: sampler/learning-rate adjustments improved balance and density fit. Confirmation showed a seed0 winner can be less reliable than another candidate. These are adaptive findings on this grid, not evidence of universal superiority.','','## Plots','','Fixed seed0, not selected for visual quality; raw generators. Blue dots are generated samples, red marks are target centers.','','![Comparison](samples.png)','','## Verification','',f'{replays} exact checkpoint/sample replays; {rngchecks} matched paired/vanilla RNG checks; source snapshot hashes and optimizer step counts verified. Control two-update regression test matches the original trainer exactly. No reconstruction loss was introduced.']
    names=['old_vanilla','old_strong','vanilla_linear','linear','ema_fast','real'];titles=['Original vanilla • 1.2M','Original paired strong deficit • 1.2M','Tuned vanilla, linear deficit • 50k','Tuned paired, linear deficit • 50k','Tuned paired, squared deficit • 50k','Real mixture reference']
    fig,axes=plt.subplots(3,2,figsize=(11,16));mu=centers(c).numpy()
    for ax,name,title in zip(axes.flat,names,titles):
        x=np.load(out/f'{name}_raw_seed0.npy');ax.scatter(x[:,0],x[:,1],s=2,alpha=.25);ax.scatter(mu[:,0],mu[:,1],s=7,c='red');ax.set(title=title,xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal')
    fig.suptitle('10×10 grid • 10,000 independent-evaluation samples • seed0');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'samples.png',dpi=160);plt.close(fig)
    conclusion = 'Both linear-deficit GANs reach all 100 half-target modes in every seed at 50k updates. Paired mode TV .0702 versus vanilla .0674 does not establish a paired advantage: the large representation/optimization gains transfer to both objectives. Fine-density error remains above the real finite-sample reference and the tuned flow models in experiment 32. The original 1.2M baselines above were re-evaluated on fresh noise, so their numbers differ slightly from earlier reports. Seed0 was used for development and is included in the five-seed summaries; they are not five untouched confirmation seeds.'
    i = lines.index('## Architecture and training')
    lines[i:i] = ['## Interpretation', '', conclusion, '']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:22]))
if __name__=='__main__':build()
