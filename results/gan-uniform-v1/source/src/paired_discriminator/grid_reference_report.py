"""Grid reference selection and adaptive real-mode reweighting."""
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
from .grid_reference import ROOT, models, coverage_metrics, base_method
from .grid_experiment import centers
from .grid_mode_sweep_report import target_grid, spatial_tv
from .grid_dual_slot_report import run_path as baseline_path

ARMS = ['paired_near','paired_deficit','vanilla_deficit']
BASE = ['vanilla','rsgan','paired','pacgan2','dual_slot']
METHODS = BASE + ARMS
DISPLAY = ['vanilla','paired'] + ARMS
METRICS = ['mode_tv','spatial_tv','valid_fraction','coverage_1pct','coverage_relative']
COLORS = {'vanilla':'#4477AA','rsgan':'#EEAA33','paired':'#228855','pacgan2':'#AA3377','dual_slot':'#00A0AA','paired_near':'#AA3377','paired_deficit':'#CC3311','vanilla_deficit':'#EEAA33'}
BASE_REPORT = ROOT/'results/grid-dual-slot-comparison-v1'


def read_csv(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('modes','seed','step','coverage_1pct','coverage_relative'):
            if k in r:r[k]=int(r[k])
        for k in ('mode_tv','spatial_tv','valid_fraction'):
            if k in r:r[k]=float(r[k])
    return rows


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def budgets(side):
    return [10000,50000]+([100000,150000] if side==7 else [])


def new_steps(side):
    return budgets(side)


def build(output):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=read_csv(BASE_REPORT/'endpoints.csv')
    histories=read_csv(BASE_REPORT/'learning_curves.csv')
    reference=json.loads((BASE_REPORT/'reference_calibration.json').read_text())
    old_hashes=json.loads((BASE_REPORT/'verification.json').read_text())['samples_sha256']
    hashes={};samples={};configs={};source_dirs=set();replays=0;rng_checks=0
    for side in (3,5,7):
        config=json.loads((ROOT/f'configs/grid{side}-{150 if side==7 else 50}k.json').read_text())
        configs[side]=config;target=target_grid(config)
        # Recheck the source samples and metrics behind the saved baseline report.
        for r in [r for r in rows if r['modes']==side*side]:
            path=ROOT/r['source'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==old_hashes[r['source']]
            hashes[r['source']]=digest
            points=np.load(path);actual=coverage_metrics(torch.from_numpy(points),config)
            assert abs(actual['mode_tv']-r['mode_tv'])<1e-10
            assert abs(actual['valid_fraction']-r['valid_fraction'])<1e-10
            assert actual['coverage']==r['coverage_1pct']
            mass=np.array([actual[f'mode_{i}_mass'] for i in range(side*side)])
            assert int((mass>=.08/(side*side)).sum())==r['coverage_relative']
            assert abs(spatial_tv(points,target)-r['spatial_tv'])<1e-10
            samples[(side,r['method'],r['seed'],r['step'])]=(points,mass)
            source_dirs.add(path.parent.parent)
        src=ROOT/f'results/grid{side}-reference-d-v1';source_dirs.add(src)
        new_config=json.loads((src/'config.json').read_text())
        assert new_config=={**config,'reference_phase':'d_only','uniform_mix':.5,'coverage_ema_decay':.99}
        for method in ARMS:
            label=method
            for seed in range(5):
                path=src/f'{method}_seed{seed}'
                with (path/'metrics.csv').open() as f:rr=list(csv.DictReader(f))
                assert [int(r['step']) for r in rr]==list(range(0,config['steps']+1,1000))
                for r in rr:
                    mass=np.array([float(r[f'mode_{i}_mass']) for i in range(side*side)])
                    histories.append(dict(modes=side*side,method=label,seed=seed,step=int(r['step']),
                        mode_tv=float(r['mode_tv']),valid_fraction=float(r['valid_fraction']),
                        coverage_1pct=int(r['coverage']),coverage_relative=int((mass>=.08/(side*side)).sum())))
                for step in new_steps(side):
                    file=path/f'samples_step{step}.npy';points=np.load(file)
                    assert points.shape==(10000,2) and np.isfinite(points).all()
                    hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
                    actual=coverage_metrics(torch.from_numpy(points),config)
                    logged=next(r for r in rr if int(r['step'])==step)
                    for k,v in actual.items():assert abs(v-float(logged[k]))<1e-10
                    saved=torch.load(path/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
                    assert saved['config']==new_config and saved['step']==step and saved['method']==method and saved['seed']==seed
                    g,d=models(config,method,seed);g.load_state_dict(saved['generator'])
                    z=torch.randn(10000,config['latent_dim'],generator=torch.Generator().manual_seed(seed+70000))
                    with torch.no_grad():assert np.array_equal(g(z).numpy(),points)
                    for optimizer in ('optimizer_d','optimizer_g'):
                        assert all(int(state['step'])==step for state in saved[optimizer]['state'].values())
                    replays+=1
                    if step in (50000,100000,150000):
                        base=baseline_path(side,base_method(method),step)/f'{base_method(method)}_seed{seed}'
                        ckpt=base/('checkpoint_100000.pt' if step==100000 else 'checkpoint.pt')
                        prior=torch.load(ckpt,map_location='cpu',weights_only=False)
                        assert prior['step']==step
                        for k in ('real_g','noise_g','slots','eval','noise_d'):
                            assert torch.equal(saved['rng_states'][k],prior['rng_states'][k])
                        if method == 'paired_near':
                            assert torch.equal(saved['rng_states']['real_d'],prior['rng_states']['real_d'])
                        rng_checks+=1
                    mass=np.array([actual[f'mode_{i}_mass'] for i in range(side*side)])
                    rows.append(dict(modes=side*side,method=label,seed=seed,step=step,
                        mode_tv=actual['mode_tv'],spatial_tv=spatial_tv(points,target),valid_fraction=actual['valid_fraction'],
                        coverage_1pct=actual['coverage'],coverage_relative=int((mass>=.08/(side*side)).sum()),source=str(file.relative_to(ROOT))))
                    samples[(side,label,seed,step)]=(points,mass)
    for src in source_dirs:
        provenance=json.loads((src/'provenance.json').read_text())
        for name,digest in provenance['source_sha256'].items():
            assert hashlib.sha256((src/'source'/name).read_bytes()).hexdigest()==digest
    assert len(rows)==320 and replays==120 and rng_checks==75
    write_csv(output/'endpoints.csv',rows);write_csv(output/'learning_curves.csv',histories)
    lookup={(r['modes'],r['method'],r['seed'],r['step']):r for r in rows}
    summaries=[];comparisons=[]
    for side in (3,5,7):
        for step in budgets(side):
            for method in METHODS:
                selected=[lookup[(side*side,method,seed,step)] for seed in range(5)]
                stats=dict(modes=side*side,step=step,method=method)
                for metric in METRICS:
                    values=[r[metric] for r in selected]
                    stats[metric]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))}
                stats['full_1pct_seeds']=sum(r['coverage_1pct']==side*side for r in selected)
                stats['full_relative_seeds']=sum(r['coverage_relative']==side*side for r in selected)
                summaries.append(stats)
            for method,other in [('paired_near','paired'),('paired_deficit','paired'),('vanilla_deficit','vanilla'),('paired_deficit','vanilla_deficit'),('paired_near','paired_deficit')]:
                for metric in ('mode_tv','spatial_tv'):
                    diffs=[lookup[(side*side,method,seed,step)][metric]-lookup[(side*side,other,seed,step)][metric] for seed in range(5)]
                    comparisons.append(dict(modes=side*side,step=step,method=method,against=other,metric=metric,
                        wins=sum(x<0 for x in diffs),mean_difference=float(np.mean(diffs)),differences=diffs))
    diagnostics=[]
    for side in (3,5,7):
        for method in ARMS:
            for seed in range(5):
                path=ROOT/f'results/grid{side}-reference-d-v1/{method}_seed{seed}/metrics.csv'
                with path.open() as f:rr=list(csv.DictReader(f))
                for step in budgets(side):
                    r=next(r for r in rr if int(r['step'])==step)
                    weights=[float(r[f'reference_weight_{i}']) for i in range(side*side)]
                    assert abs(sum(weights)-1)<1e-10 and min(weights)>=.5/(side*side)-1e-10
                    before=float(r['d_pair_rms_before_mean']);after=float(r['d_pair_rms_after_mean'])
                    if method=='paired_near':assert after<=before+1e-10 and before>0
                    diagnostics.append(dict(modes=side*side,method=method,seed=seed,step=step,
                        rms_distance_before=before,rms_distance_after=after,reference_weights=weights,
                        ema_mass=[float(r[f'ema_mass_{i}']) for i in range(side*side)]))
    audit=json.loads((ROOT/'results/grid-reference-role-audit-v1/audit.json').read_text())
    for name,data in [('summary.json',summaries),('paired_comparisons.json',comparisons),('reference_calibration.json',reference),
                      ('reference_diagnostics.json',diagnostics),('source_role_audit.json',audit),
                      ('verification.json',dict(verified_endpoints=320,new_checkpoint_replays=120,g_and_noise_d_rng_checks=75,
                       reused_baseline_endpoints=200,new_trajectories=45,samples_sha256=hashes))]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for j,side in enumerate((3,5,7)):
        for i,metric in enumerate(('mode_tv','valid_fraction')):
            ax=axes[i,j]
            for method in DISPLAY:
                curves=[]
                for seed in range(5):
                    rr=sorted([r for r in histories if (r['modes'],r['method'],r['seed'])==(side*side,method,seed)],key=lambda r:r['step'])
                    x=np.array([r['step'] for r in rr])/1000
                    y=np.array([r[metric] for r in rr]);curves.append(y)
                array=np.array(curves);mean=array.mean(0);sd=array.std(0,ddof=1)
                ax.plot(x,mean,color=COLORS[method],label=method)
                ax.fill_between(x,mean-sd,mean+sd,color=COLORS[method],alpha=.1)
            ax.set_ylim(0,1);ax.set_title(f'{side}×{side} · '+('Mode TV ↓' if i==0 else 'Valid fraction ↑'))
            ax.set_xlabel('G updates (thousands)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('D-only reference interventions · unchanged G training · five seeds, mean ± sample SD')
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for side,ax in zip((3,5,7),axes):
        for method in DISPLAY:
            rr=[r for r in summaries if r['modes']==side*side and r['method']==method]
            ax.errorbar([r['step']/1000 for r in rr],[r['spatial_tv']['mean'] for r in rr],yerr=[r['spatial_tv']['sd'] for r in rr],
                color=COLORS[method],marker='o',capsize=3,label=method)
        ax.axhline(reference[str(side*side)]['real_spatial_tv_mean'],color='black',ls='--',label='Real-sample reference')
        ax.set_ylim(0,1);ax.set_title(f'{side}×{side} · Fine-grid TV ↓');ax.set_xlabel('G updates (thousands)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Density fit against the original uniform mixture · mean ± sample SD')
    fig.tight_layout();fig.savefig(output/'density_tv.png',dpi=160);plt.close(fig)
    for side in (3,5,7):
        for step in budgets(side):
            fig,axes=plt.subplots(5,5,figsize=(13,13),sharex=True,sharey=True)
            ctr=centers(configs[side]).numpy();limit=(side-1)/2*1.5+.7
            for j,method in enumerate(DISPLAY):
                for seed in range(5):
                    ax=axes[j,seed];points,_=samples[(side,method,seed,step)]
                    ax.scatter(points[:3000,0],points[:3000,1],s=.7,alpha=.35,color=COLORS[method],rasterized=True)
                    ax.scatter(ctr[:,0],ctr[:,1],marker='x',s=8,color='black')
                    ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal')
                    outside=np.any(np.abs(points[:3000])>limit,axis=1).mean()
                    if outside>.001:ax.text(.02,.02,f'Outside view: {outside:.1%}',transform=ax.transAxes,fontsize=6,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                    if seed==0:ax.set_ylabel(method)
                    if j==0:ax.set_title(f'Seed {seed}')
            fig.suptitle(f'{side}×{side} · {step//1000}k · first 3,000 fixed generated samples · × = real mode centers')
            fig.tight_layout();fig.savefig(output/f'samples_grid{side}_{step//1000}k.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(17,10),layout='constrained')
    for i,side in enumerate((3,5,7)):
        step=150000 if side==7 else 50000
        for j,method in enumerate(DISPLAY):
            ratio=np.mean([samples[(side,method,seed,step)][1] for seed in range(5)],axis=0)*side*side
            ax=axes[i,j];im=ax.imshow(np.log2(np.clip(ratio,1/16,16)).reshape(side,side),origin='lower',cmap='RdBu_r',vmin=-4,vmax=4)
            ax.set_title(f'{side}×{side} · {method}',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,ticks=[-4,-2,0,2,4]);cb.ax.set_yticklabels(['≤1/16×','1/4×','1× target','4×','≥16×'])
    fig.suptitle('Final accepted mode mass / uniform target mass · five-seed mean')
    fig.savefig(output/'mode_mass_maps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,method in enumerate(('paired_deficit','vanilla_deficit')):
        for j,side in enumerate((3,5,7)):
            step=150000 if side==7 else 50000
            values=np.array([r['reference_weights'] for r in diagnostics if (r['modes'],r['method'],r['step'])==(side*side,method,step)])*side*side
            ax=axes[i,j];ax.bar(range(side*side),values.mean(0),color=COLORS[method]);ax.errorbar(range(side*side),values.mean(0),yerr=values.std(0,ddof=1),fmt='none',ecolor='black',lw=.6)
            ax.axhline(1,color='black',ls='--');ax.axhline(.5,color='gray',ls=':')
            ax.set_title(f'{side}×{side} · {method}');ax.set_xlabel('Mode index');ax.set_ylabel('Real sampling probability / uniform')
    fig.suptitle('Final adaptive D real sampling weights · mean ± sample SD · floor = 0.5× uniform')
    fig.tight_layout();fig.savefig(output/'sampling_weights.png',dpi=160);plt.close(fig)
    write_report(output,summaries,comparisons,audit,diagnostics)
    source=output/'source';source.mkdir(exist_ok=True);provenance={}
    for name in ('grid_reference.py','grid_reference_run.py','grid_reference_report.py','grid_reference_audit.py','grid_pacgan.py','grid_experiment.py','grid_mode_sweep_report.py','grid_dual_slot_report.py'):
        p=ROOT/'src/paired_discriminator'/name;shutil.copyfile(p,source/name);provenance[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copyfile(ROOT/'docs/GRID_REFERENCE_PROTOCOL.md',output/'PROTOCOL.md')
    (output/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def write_report(output,summaries,comparisons,audit,diagnostics):
    lines=['# Nearby and underrepresented-mode references on Gaussian grids','',
        '45 new runs: paired_near, paired_deficit and vanilla_deficit × seeds 0–4 × 3×3/5×5/7×7. Only D real sampling/pairing changes; G references stay random and uniform. Original D256/G256, architectures, optimizer settings and budgets. Smaller grids run to 50k; 7×7 to 150k. Existing random paired/vanilla and other historical methods are reused. Sample and update counts match, but exact matching adds substantial CPU work; compute is not matched. [Prespecified protocol](PROTOCOL.md).','',
        '## Findings','',
        'The useful signal comes from emphasizing underrepresented modes, not from the D-only nearby matching tested here. Adaptive sampling improves final mean mode TV for vanilla on every grid and for paired on 5×5/7×7, but destabilizes one paired 3×3 seed. Neither policy is a universal improvement.\n\nNearby paired performs poorly: final mean mode TV is 0.944 on 3×3, 0.982 on 5×5, 0.963 on 7×7. The corresponding random-paired scores are 0.058, 0.242, 0.231. The matching diagnostics confirm reduced pair distances and the real-real audit is consistent with chance source-role prediction. However, D learns on matched pairs while G uses independent random references; the result may reflect that context mismatch and does not reject nearby pairing in both phases. One-to-one matching also cannot make every reference local after G collapses, because it must retain the full real batch.\n\nOn 7×7 at 150k, adaptive paired improves mode TV 0.231→0.183 and fine-grid TV 0.647→0.573. It wins those metrics in 4/5 and 4/5 matched seeds, respectively. Original-threshold coverage improves 41.4→49.0 of 49 modes; all 49 clear that threshold in all five adaptive-paired seeds, versus one original-paired seed. Adaptive vanilla also benefits substantially: mode TV 0.520→0.320, fine-grid TV 0.723→0.681. Thus an appreciable part of the benefit comes from the real-sampling rule itself, not uniquely from joint inputs. Paired remains ahead of the adaptive vanilla control at this endpoint. Both remain above the real-sample fine-grid reference of 0.264.\n\nOn 5×5 at 50k, adaptive paired improves mode TV 0.242→0.214, with all 25 modes above the original 1% threshold in all five seeds. But fine-grid TV slightly worsens (0.597→0.606). Adaptive vanilla improves mode TV 0.718→0.603 while remaining far behind paired. On 3×3, adaptive vanilla improves mode allocation (two seeds reach six modes rather than four), without improving mean fine-grid TV. Adaptive paired has four full-coverage seeds and one catastrophic failure: seed 1 has zero valid samples at 50k, raising mean mode TV to 0.245 versus 0.058. Generated coordinates and losses remain finite; the failure is retained, not excluded.\n\nInterpretation: prioritizing missing modes is a promising synthetic-data intervention, with meaningful benefits beyond ordinary paired training on 7×7. It uses known Gaussian geometry and changes D’s real training distribution, so it is not a pure reference-association test or a ready-made image method. The vanilla control uses the same algorithm driven by its own G, not the same realized weights. Paired-deficit also retains uniform G references, introducing a D/G context difference. No mixture strength or EMA tuning was performed, and the seed failure and mixed density rankings matter. Further tests could isolate the phase choice, but none were added to this run.','',
        '## Policies and controls','',
        '**Nearby paired:** minimum-sum squared-Euclidean-distance one-to-one matching of the original uniform real batch to the generated batch. Every point appears once, both marginals remain unchanged, fake order is retained, and A/B slots are randomized as before. Matching biases the association; it does not resample easier real points. D trains on near pairs, while G trains with random pairs, so their pair-context distributions differ.','',
        '**Underrepresented-mode sampling:** use the known Gaussian centers and acceptance radius to track accepted generator mass q_k with an EMA (decay 0.99, initialized at 1/K). Each step uses its previous EMA to set d_k=max(1/K−q_k,0), then samples real modes with w_k=0.5/K+0.5d_k/Σd. Use uniform fallback when all deficits vanish. Fresh Gaussian noise retains sigma 0.1. Update the estimate from the existing D fake batch; no extra G samples or evaluation-noise feedback. Invalid mass remains in the denominator. Every mode has probability at least 0.5/K. Equal accepted mass across modes produces uniform weights; this does not guarantee stability. Log estimates and weights.','',
        '**Vanilla control:** applies the same adaptive rule to its own generator. This controls the sampling algorithm, not an identical weight trajectory. The deficit policy changes the real training distribution, unlike nearby matching. Paired-deficit also has a context difference: G’s references remain uniform while D’s real samples are weighted. Metrics always evaluate against the original uniform mixture. Differences between near and deficit arms therefore do not isolate pairing distance.','',
        '## Real-versus-real diagnostic','',
        'Nearby matching uses symmetric costs; swapping source batches produces the same matched edges in the structural test. A separate learned diagnostic trains D for 2,000 steps on two independent real source batches (5×5, three seeds), then evaluates 51,200 fresh pairs per seed. Source-role accuracy should be 50%. Standard errors below are across 200 batches, allowing within-batch dependence.','',
        '| Seed | Held-out accuracy | Batch-based SE |','|---:|---:|---:|']
    for r in audit['results']:lines.append(f"| {r['seed']} | {r['heldout_accuracy']:.2%} | {r['batch_accuracy_se']:.2%} |")
    lines+=['','The diagnostic is consistent with chance; it does not prove absence of every possible source-role signal. It applies to nearby matching, not adaptive reweighting, whose purpose is to change the real sampling distribution.','',
        '## Metrics and results','',
        'Mean ± sample SD across five seeds. Mode TV = ½(Σ|accepted mass−1/K|+invalid mass), radius 0.3, all generated samples in denominator. Original coverage requires ≥1% of all samples per mode; relative coverage requires ≥8% of ideal 1/K mass. Fine-grid TV measures density using exact mixture probabilities in 0.05-wide cells over [-5.5,5.5]² plus an outside category. Real-sample calibration accounts for sampling noise, not training uncertainty.','',
        '| Modes | Updates | Method | Mode TV ↓ | Fine-grid TV ↓ | Valid | Coverage ≥1% | Relative coverage |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
        lines.append(f"| {r['modes']} | {r['step']//1000}k | {r['method']} | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['valid_fraction']['mean']:.1%} | {r['coverage_1pct']['mean']:.1f}/{r['modes']} | {r['coverage_relative']['mean']:.1f}/{r['modes']} |")
    lines+=['','![Learning curves](training_curves.png)','','![Density TV](density_tv.png)','','![Mode mass](mode_mass_maps.png)','','![Sampling weights](sampling_weights.png)','','## Matching diagnostics','','Cumulative mean batch RMS distance before/after matching at the final endpoint, averaged across five seeds. This is a geometric check, not a distribution-quality metric.','','| Grid | Random pairing RMS | Matched RMS |','|---|---:|---:|']
    for side in (3,5,7):
        rr=[r for r in diagnostics if (r['modes'],r['method'],r['step'])==(side*side,'paired_near',150000 if side==7 else 50000)]
        lines.append(f"| {side}×{side} | {np.mean([r['rms_distance_before'] for r in rr]):.3f} | {np.mean([r['rms_distance_after'] for r in rr]):.3f} |")
    lines+=['','## All-seed samples','']
    for side in (3,5,7):
        for step in budgets(side):lines.append(f'- [{side}×{side}, {step//1000}k](samples_grid{side}_{step//1000}k.png)')
    lines+=['','## Verification and reproduction','',
        'All 320 retained endpoints verified from saved arrays, including 120 new endpoints that regenerate bit-for-bit from checkpoints and 200 reused endpoints whose hashes match the prior verified report. Source hashes, configurations and optimizer step counts checked. G real/noise/slot/evaluation and D-noise RNG states match original baselines at all 75 mutually retained checkpoints. The real sampling stream also matches for nearby arms and changes for deficit arms as intended. Diagnostics verify normalized weights, the uniform floor and reduced near-pair distance.','',
        'Run `uv run python -m paired_discriminator.grid_reference_audit`, then `uv run python -m paired_discriminator.grid_reference_run`, then `uv run python -m paired_discriminator.grid_reference_report results/grid-reference-comparison-v1`. Launchers refuse existing outputs. SciPy provides exact assignment; dependencies are locked. Full matched-seed differences, source, sample arrays and metrics are retained. Checkpoints remain local and ignored by Git.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    import sys
    build(sys.argv[1])
