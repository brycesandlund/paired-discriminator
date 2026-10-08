"""Consolidate the adaptive study without rerunning training or evaluation."""
import json,platform,re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_experiment import ROOT,centers

def main():
    out=ROOT/'results/flow-ablation-comparison-v1';base=json.loads((out/'validation_metrics.json').read_text())
    refine=json.loads((out/'validation_metrics_refine.json').read_text());wide=json.loads((out/'validation_metrics_wide256.json').read_text())
    assert len(refine)==len(wide)==10
    allrows=base+refine+wide
    summary=[]
    labels={'analytic':'Analytic oracle','two_layers':'Tuned 2×128, 50k','cosine':'Tuned 3×128, 50k','refine':'Tuned 3×128, 100k','wide256':'Tuned 3×256, 50k'}
    for name in labels:
        for steps in [128,256]:
            rr=[r for r in allrows if r['name']==name and r['solver_steps']==steps];assert len(rr)==5
            summary.append(dict(name=name,solver_steps=steps,**{k:dict(mean=float(np.mean([r[k] for r in rr])),sd=float(np.std([r[k] for r in rr],ddof=1))) for k in ['mode_tv','spatial_tv','valid_fraction','coverage_half_target','coverage_relative','sampling_seconds']}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,python=platform.python_version(),platform=platform.platform()),indent=2)+'\n')
    head=['# Flow matching tuning: near-analytic sample quality','','The learned model now resolves all 100 modes after 50k updates. The analytic oracle remains slightly better on aggregate mass allocation; agreement is close on these measured distributional metrics, not an assertion of identical fields. No oracle targets, mode identities, grid spacing or component variance enter the confirmed learned models.','','## Final comparison','','Five-seed means on independent 10k-sample noise per seed, midpoint128. The exact same fresh noise is used across the tuned models and analytic oracle.','','| Model | Mode TV ↓ | Fine TV ↓ | Valid mass ↑ | Half-target modes ↑ |','|---|---:|---:|---:|---:|']
    for r in summary:
        if r['solver_steps']==128:head.append(f"| {labels[r['name']]} | {r['mode_tv']['mean']:.4f} | {r['spatial_tv']['mean']:.4f} | {r['valid_fraction']['mean']:.2%} | {r['coverage_half_target']['mean']:.1f}/100 |")
    head+=['','![Final samples](final_samples.png)','','All learned models in this table reach half-target coverage 100/100 on every seed. See summary.json for SDs and solver256 results. Original flow (#31) at 1.2M was mode TV .6166, fine TV .7753, valid mass 38.45%, half-target coverage 24.8/100, evaluated with the earlier fixed-noise seeds.','','## What the search established','','1. Generic Fourier position/time features were the largest observed improvement in the controlled seed-0 screen. Increasing depth or adjusting the optimizer alone was much less effective at 20k.','2. Scaling coordinates by their expected time-dependent spread and predicting a residual around a Gaussian base field improved the Fourier model further. Only a data standard deviation estimated from a 100k-sample pilot is needed.','3. Cosine learning-rate decay stabilizes final quality. Two hidden layers already work; width256 brings valid mass closer to the analytic field.','4. Low-rate refinement of width128 modestly improves mode TV, without much change in valid mass. Higher solver resolution has negligible effect.','','This is an adaptive, problem-specific tuning study. Seed 0 drove initial selection; other seeds confirmed the initial choice. Later refinement and wider-model confirmation are additional adaptive steps. It is not a benchmark-wide claim or a matched-parameter/compute GAN comparison. Generation still costs 256 network evaluations at midpoint128.','','## Follow-up validation','','All five 50k→100k resume boundaries and hashes passed. All 20 final learned checkpoints (two initial architectures, refinement, wider architecture) replay saved samples exactly. Optimizer step counts and source snapshots verified. Three focused analytic/feature tests passed. Raw weights are used throughout the final table; EMA outcomes remain in the full trial logs.','','31 short training jobs total: 8 screening, 6 refinement screens, 8 initial confirmation, 5 low-rate continuations, 4 wider-model confirmation. Two screening jobs use oracle supervision and are diagnostic only. All trial logs and checkpoints retained.','','## Reproduction','','`uv run python -m paired_discriminator.flow_ablation` (round 1)  ','`uv run python -m paired_discriminator.flow_ablation_r2` (round 2)  ','`uv run python -m paired_discriminator.flow_ablation_validate` (initial confirmation)  ','`uv run python -m paired_discriminator.flow_ablation_refine` (low-rate continuation)  ','`uv run python -m paired_discriminator.flow_ablation_wide` (wider confirmation)','','Commands intentionally refuse to overwrite result directories. Full configurations and frozen source accompany each experiment.','','---','']
    head += ['## Model size and measured training cost','','| Model | Parameters | Cumulative train seconds (mean) | Real draws per run |','|---|---:|---:|---:|']
    for name in ['two_layers','cosine','refine','wide256']:
        metas=[]
        for seed in range(5):
            folder='flow-ablation-refine-v1' if name=='refine' else 'flow-ablation-r2-v1' if seed==0 else 'flow-ablation-wide-validation-v1' if name=='wide256' else 'flow-ablation-validation-v1'
            metas.append(json.loads((ROOT/'results'/folder/f'{name}_seed{seed}'/'metadata.json').read_text()))
        head.append(f"| {labels[name]} | {metas[0]['parameters']} | {np.mean([v['training_seconds'] for v in metas]):.2f} | {metas[0]['real_samples']:,} + shared 100k pilot |")
    head += ['','Timing depends on concurrent workloads. Higher model capacity and sampling cost are part of the tradeoff.','']
    trialfile=out/'all_trials.json';trials=json.loads(trialfile.read_text())
    for folder in ['flow-ablation-refine-v1','flow-ablation-wide-validation-v1']:
        for path in (ROOT/'results'/folder).glob('*/metrics.json'):
            for r in json.loads(path.read_text()):
                row=dict(round=folder,oracle=False,**r)
                if not any(v['round']==folder and v['name']==r['name'] and v['seed']==r['seed'] and v['step']==r['step'] and v['variant']==r['variant'] for v in trials):trials.append(row)
    trialfile.write_text(json.dumps(trials,indent=2)+'\n')
    old=(out/'REPORT.md').read_text()
    if '\n---\n' in old and old.startswith('# Flow matching tuning:'):old=old.split('\n---\n',1)[1]
    (out/'REPORT.md').write_text('\n'.join(head)+old)
    c=json.loads((ROOT/'results/flow-ablation-r2-v1/config.json').read_text());mu=centers(c).numpy()
    paths=[ROOT/'results/grid10-flow-1200k-v1/seed0/samples_1200000_solver128.npy',out/'analytic_seed0_solver128.npy',out/'two_layers_seed0_solver128.npy',out/'wide256_seed0_solver128.npy']
    titles=['Original learned flow • 1.2M','Exact analytic field','Tuned 2×128 flow • 50k','Tuned 3×256 flow • 50k']
    fig,axes=plt.subplots(2,2,figsize=(11,11))
    for ax,path,title in zip(axes.flat,paths,titles):
        x=np.load(path);ax.scatter(x[:,0],x[:,1],s=2,alpha=.25);ax.scatter(mu[:,0],mu[:,1],s=7,c='red');ax.set(title=title,xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal')
    fig.suptitle('10×10 grid • 10,000 samples • seed 0 • midpoint128');fig.tight_layout();fig.savefig(out/'final_samples.png',dpi=170);plt.close(fig)
    p=ROOT/'training_runs.md';s=p.read_text();table,rest=s.split('## What the methods mean',1)
    table=re.sub(r'(?<=\|)\n\s*\n(?=\| \d+ \|)','\n',table)
    row='| 32 | Flow-matching tuning against analytic reference | 31 short jobs; five-seed validation of sample-only models | 20k screens, 50k confirmation, 100k refinement | Complete: tuned flow reaches all 100 half-target modes in every seed; near-analytic mass/density metrics. |\n'
    if '| 32 | Flow-matching tuning' not in table:table=table.rstrip()+'\n'+row+'\n'
    s=table+'## What the methods mean'+rest
    section=['## 32. Flow matching tuning against the analytic reference','','[Protocol](docs/FLOW_ABLATIONS_PROTOCOL.md) · [Report](results/flow-ablation-comparison-v1/REPORT.md).','','31 short jobs across screening, confirmation, and refinement. Two oracle-teacher trials are diagnostics only. Final learned models train on ordinary independent noise/real pairs with velocity MSE; no mixture parameters or labels. The pilot estimates only data scale. Fixed generic Fourier features, Gaussian scale/residual preconditioning, Adam (.9,.999), and cosine lr .001→.0001 over 50k make the major improvement. Batch256. Both two- and three-hidden-layer networks work.','','Five-seed fresh-noise means at midpoint128:','','| Model | Mode TV | Fine TV | Valid mass | Half-target coverage |','|---|---:|---:|---:|---:|']
    for r in summary:
        if r['solver_steps']==128:section.append(f"| {labels[r['name']]} | {r['mode_tv']['mean']:.4f} | {r['spatial_tv']['mean']:.4f} | {r['valid_fraction']['mean']:.2%} | {r['coverage_half_target']['mean']:.1f}/100 |")
    section+=['','All tuned models cover 100/100 half-target modes in every seed. Raw weights fixed before confirmation. Seed0 development, seeds1–4 initial confirmation; follow-up decisions were adaptive. Midpoint256 barely changes results. Original #30–31 flow results were an under-tuned setup and do not represent a competitive flow baseline.','','20 final checkpoint/sample replays, all source hashes, optimizer steps, and five refinement resume boundaries passed; three focused tests passed. No claim of exact field equality everywhere. Results are not matched-parameter or matched-compute GAN comparisons. Checkpoints, all trial outcomes and plots preserved; no commits made.']
    if '\n## 32. Flow matching tuning against the analytic reference' in s:s=s.split('\n## 32. Flow matching tuning against the analytic reference')[0]
    p.write_text(s.rstrip()+'\n\n'+'\n'.join(section)+'\n')
    print('\n'.join(head[:19]))
if __name__=='__main__':main()
