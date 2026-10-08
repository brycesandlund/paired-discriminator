"""D512 versus D256, at equal G updates and equal cumulative D sample counts."""
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
from .grid_d_batch import ROOT, models, coverage_metrics
from .grid_experiment import centers
from .grid_mode_sweep_report import target_grid, spatial_tv
from .grid_dual_slot_report import run_path as baseline_path

JOINT = ['paired', 'pacgan2', 'dual_slot']
BASE = ['vanilla', 'rsgan'] + JOINT
METHODS = BASE + [m+'_d512' for m in JOINT]
METRICS = ['mode_tv','spatial_tv','valid_fraction','coverage_1pct','coverage_relative']
COLORS = {'vanilla':'#4477AA','rsgan':'#EEAA33','paired':'#228855','pacgan2':'#AA3377','dual_slot':'#00A0AA'}
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
    return sorted(set(budgets(side)+[s//2 for s in budgets(side)]))


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
        src=ROOT/f'results/grid{side}-d512-v1';source_dirs.add(src)
        new_config=json.loads((src/'config.json').read_text())
        assert new_config=={**config,'discriminator_batch_size':512}
        for method in JOINT:
            label=method+'_d512'
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
                        base=baseline_path(side,method,step)/f'{method}_seed{seed}'
                        if method in ('pacgan2','dual_slot'):ckpt=base/f'checkpoint_{step}.pt'
                        else:ckpt=base/('checkpoint_100000.pt' if step==100000 else 'checkpoint.pt')
                        prior=torch.load(ckpt,map_location='cpu',weights_only=False)
                        assert prior['step']==step
                        for k in ('real_g','noise_g','slots','eval'):
                            assert torch.equal(saved['rng_states'][k],prior['rng_states'][k])
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
    assert len(rows)==425 and replays==225 and rng_checks==75
    write_csv(output/'endpoints.csv',rows);write_csv(output/'learning_curves.csv',histories)
    lookup={(r['modes'],r['method'],r['seed'],r['step']):r for r in rows}
    summaries=[];comparisons=[]
    for accounting in ('equal_g_updates','equal_d_samples'):
        for side in (3,5,7):
            for budget in budgets(side):
                for method in METHODS:
                    step=budget//2 if accounting=='equal_d_samples' and method.endswith('_d512') else budget
                    selected=[lookup[(side*side,method,seed,step)] for seed in range(5)]
                    stats=dict(accounting=accounting,modes=side*side,reference_steps=budget,method=method,actual_steps=step,
                        d_input_points=step*(1024 if method.endswith('_d512') else 512),g_training_points=step*256)
                    for metric in METRICS:
                        values=[r[metric] for r in selected]
                        stats[metric]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))}
                    stats['full_1pct_seeds']=sum(r['coverage_1pct']==side*side for r in selected)
                    stats['full_relative_seeds']=sum(r['coverage_relative']==side*side for r in selected)
                    summaries.append(stats)
                for method in JOINT:
                    for other in (method,'vanilla','rsgan'):
                        step=budget//2 if accounting=='equal_d_samples' else budget
                        for metric in ('mode_tv','spatial_tv'):
                            diffs=[lookup[(side*side,method+'_d512',seed,step)][metric]-lookup[(side*side,other,seed,budget)][metric] for seed in range(5)]
                            comparisons.append(dict(accounting=accounting,modes=side*side,reference_steps=budget,
                                method=method+'_d512',against=other,metric=metric,wins=sum(x<0 for x in diffs),
                                mean_difference=float(np.mean(diffs)),differences=diffs))
    for name,data in [('summary.json',summaries),('paired_comparisons.json',comparisons),('reference_calibration.json',reference),
                      ('verification.json',dict(verified_endpoints=425,new_checkpoint_replays=225,g_rng_checks=75,
                       reused_baseline_endpoints=200,new_trajectories=45,samples_sha256=hashes))]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    plot_curves(output,histories)
    plot_endpoints(output,summaries,reference)
    for side in (3,5,7):
        final=150000 if side==7 else 50000
        for step in (10000,final):
            fig,axes=plt.subplots(6,5,figsize=(13,16),sharex=True,sharey=True)
            ctr=centers(configs[side]).numpy();limit=(side-1)/2*1.5+.7
            for j,method in enumerate(JOINT):
                for variant,label in enumerate((method,method+'_d512')):
                    for seed in range(5):
                        ax=axes[j*2+variant,seed];points,_=samples[(side,label,seed,step)]
                        ax.scatter(points[:3000,0],points[:3000,1],s=.7,alpha=.35,color=COLORS[method],rasterized=True)
                        ax.scatter(ctr[:,0],ctr[:,1],marker='x',s=7,color='black')
                        ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal')
                        if seed==0:ax.set_ylabel(label)
                        if j==variant==0:ax.set_title(f'Seed {seed}')
            fig.suptitle(f'{side}×{side} · {step//1000}k G updates · 3,000 fixed samples · D256 versus D512')
            fig.tight_layout();fig.savefig(output/f'samples_grid{side}_{step//1000}k.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,6,figsize=(19,10),layout='constrained')
    for i,side in enumerate((3,5,7)):
        step=150000 if side==7 else 50000
        for j,method in enumerate(JOINT):
            for variant,label in enumerate((method,method+'_d512')):
                ax=axes[i,j*2+variant]
                ratio=np.mean([samples[(side,label,seed,step)][1] for seed in range(5)],axis=0)*side*side
                im=ax.imshow(np.log2(np.clip(ratio,1/16,16)).reshape(side,side),origin='lower',cmap='RdBu_r',vmin=-4,vmax=4)
                ax.set_title(f'{side}×{side} · {label}',fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,ticks=[-4,-2,0,2,4]);cb.ax.set_yticklabels(['≤1/16×','1/4×','1× target','4×','≥16×'])
    fig.suptitle('Final accepted mode mass / ideal mass · five-seed mean · equal G updates')
    fig.savefig(output/'mode_mass_maps.png',dpi=160);plt.close(fig)
    write_report(output,summaries,comparisons)
    source=output/'source';source.mkdir(exist_ok=True);hashes={}
    for name in ('grid_d_batch.py','grid_d_batch_run.py','grid_d_batch_report.py','grid_dual_slot.py','grid_pacgan.py','grid_experiment.py','grid_mode_sweep_report.py','grid_dual_slot_report.py'):
        p=ROOT/'src/paired_discriminator'/name;shutil.copyfile(p,source/name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'analysis_provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')


def plot_curves(output,histories):
    for accounting in ('equal_g_updates','equal_d_samples'):
        fig,axes=plt.subplots(3,3,figsize=(15,12))
        for i,method in enumerate(JOINT):
            for j,side in enumerate((3,5,7)):
                ax=axes[i,j]
                for label,color,style in [('vanilla',COLORS['vanilla'],':'),('rsgan',COLORS['rsgan'],':'),(method,COLORS[method],'--'),(method+'_d512',COLORS[method],'-')]:
                    curves=[]
                    for seed in range(5):
                        rr=sorted([r for r in histories if (r['modes'],r['method'],r['seed'])==(side*side,label,seed)],key=lambda r:r['step'])
                        multiplier=(2 if label.endswith('_d512') else 1) if accounting=='equal_d_samples' else 1
                        x=np.array([r['step'] for r in rr])*multiplier*(512/1e6 if accounting=='equal_d_samples' else 1/1000)
                        values=np.array([r['mode_tv'] for r in rr]);curves.append(values)
                    array=np.array(curves);mean=array.mean(0);sd=array.std(0,ddof=1)
                    ax.plot(x,mean,color=color,ls=style,label=label,lw=1.7)
                    ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.09)
                maximum=150000 if side==7 else 50000
                ax.set_xlim(0,maximum*(512/1e6 if accounting=='equal_d_samples' else 1/1000));ax.set_ylim(0,1);ax.grid(alpha=.2)
                ax.set_title(f'{side}×{side} · {method}');ax.set_ylabel('Mode TV ↓')
                ax.set_xlabel('G updates (thousands)' if accounting=='equal_g_updates' else 'Total D input points (millions)')
                if j==0:ax.legend(fontsize=8)
        title='Equal G updates: D512 sees twice the D samples' if accounting=='equal_g_updates' else 'Equal D samples: D512 has half as many optimizer updates'
        fig.suptitle(title+'\nFive seeds · mean ± sample SD · dashed D256 / solid D512')
        fig.tight_layout();fig.savefig(output/f'curves_{accounting}.png',dpi=160);plt.close(fig)


def plot_endpoints(output,summaries,reference):
    for accounting in ('equal_g_updates','equal_d_samples'):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for j,side in enumerate((3,5,7)):
            for i,metric in enumerate(('mode_tv','spatial_tv')):
                ax=axes[i,j]
                for method in METHODS:
                    rr=[r for r in summaries if r['accounting']==accounting and r['modes']==side*side and r['method']==method]
                    base=method.removesuffix('_d512');color=COLORS[base]
                    style='-' if method.endswith('_d512') else ('--' if base in JOINT else ':')
                    ax.errorbar([r['reference_steps']/1000 for r in rr],[r[metric]['mean'] for r in rr],
                        yerr=[r[metric]['sd'] for r in rr],color=color,ls=style,marker='o',markersize=3,capsize=2,label=method,alpha=.9)
                if metric=='spatial_tv':ax.axhline(reference[str(side*side)]['real_spatial_tv_mean'],color='black',ls='--',lw=.8)
                ax.set_title(f'{side}×{side} · '+('Mode TV ↓' if i==0 else 'Fine-grid TV ↓'))
                ax.set_xlabel('Reference D256 steps (thousands)');ax.set_ylim(0,1);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,ncol=2)
        fig.suptitle(accounting.replace('_',' ')+' · mean ± sample SD · solid = D512; dashed = joint D256')
        fig.tight_layout();fig.savefig(output/f'endpoints_{accounting}.png',dpi=160);plt.close(fig)


def write_report(output,summaries,comparisons):
    lines=['# Doubling only discriminator batch on Gaussian grids','',
        '45 new trajectories: paired, PacGAN2 and two-output × seeds 0–4 × 3×3/5×5/7×7. D batch increases from 256 real + 256 generated to 512 real + 512 generated points. G continues to train on 256 fresh generated points with each method’s original objective and reference pairing. Same architectures, initial weights, learning rates, Adam betas, grid geometry and one D/G update per step. 3×3/5×5 run to 50k; 7×7 to 150k. Existing five-method baselines are reused. Eight local CPU workers, one torch thread each.','',
        '## Findings','',
        'Doubling D’s batch does not remove the joint models’ characteristic early disadvantage or produce a reliable speedup. There are modest early improvements on the larger grids, but no consistent benefit at the final budgets. The hypothesis that simply giving D more examples per update would fix the lag is not supported by this intervention.\n\nAt equal G updates on 5×5 at 10k, D512 improves mean mode TV from 0.728→0.691 (paired), 0.741→0.696 (PacGAN2), and 0.743→0.703 (two-output), but all remain behind vanilla/RSGAN at 0.657/0.659. Original-threshold coverage is 13.6/14.2/13.6 modes, versus 15.2/14.6 for unary methods. On 7×7 at 50k there is a qualification: paired D512 reaches 0.659 mode TV, slightly better than vanilla/RSGAN at 0.667/0.668, while PacGAN2/two-output remain behind at 0.692/0.704. Paired’s fine-grid TV and original-threshold coverage still trail the unary baselines there. Thus the early gap narrows in some settings rather than remaining completely unchanged.\n\nAt the final budgets, all three D512 arms have worse mean mode TV than their own D256 baselines on both 5×5 and 7×7. At 5×5/50k the changes are 0.242→0.258 paired, 0.216→0.271 PacGAN2, and 0.211→0.243 two-output; each loses in 4/5 matched seeds. At 7×7/150k they are 0.231→0.286, 0.229→0.248, and 0.209→0.281. Fine-grid TV at that endpoint changes from 0.647→0.696, 0.655→0.656, and 0.595→0.665; paired and two-output lose on fine-grid TV in all five matched seeds. Original-threshold coverage falls from 41.4→35.2, 43.2→38.6, and 44.8→34.6 of 49 modes. All three still beat vanilla/RSGAN on mean final mode TV. On 3×3 the result is mixed and all runs cover all nine modes; PacGAN2 improves mean fine-grid TV while two-output worsens it.\n\nAt equal cumulative D samples, mean mode TV is worse with D512 in all 24 reported method/grid/reference-budget combinations versus the same D256 method. This comparison gives D512 half as many G/D optimizer updates, so it tests sample efficiency rather than equal optimization effort. For the 7×7 budget corresponding to D256 at 150k, D512 runs 75k and scores 0.507/0.512/0.529, versus 0.231/0.229/0.209 for its D256 counterparts. More D examples in fewer updates do not substitute for the original training trajectory.\n\nInterpretation: the slower start and stronger late mode allocation of joint models largely survive the intervention; extra D batch data does not explain them away. A more difficult learned function remains a plausible hypothesis, not an established mechanism. Batch size changes gradient noise and optimization dynamics while leaving representational capacity fixed; this result does not rule out capacity limitations. The same learning rate was intentionally retained, so these findings concern this controlled change rather than the best achievable larger-batch configuration. These are finite-budget endpoints, not demonstrated convergence plateaus, and none establishes full density recovery.\n','',
        '## Workloads and random streams','',
        '| Per optimizer update | Paired D512 | PacGAN2 D512 | Two-output D512 |',
        '|---|---:|---:|---:|',
        '| D real / generated points | 512 / 512 | 512 / 512 | 512 / 512 |',
        '| D input pairs | 512 RF/FR | 256 RR + 256 FF | 128 each RR/RF/FR/FF |',
        '| D BCE decisions | 512 | 512 | 1,024 |',
        '| G generated points | 256 | 256 | 256 |',
        '| G real references used | 256 | 0 | 128 |',
        '| D input pairs during G update | 256 | 128 | 192 |','',
        'Losses retain mean reduction; D receives one larger update, not two smaller updates. G’s optimizer batch stays at 256, but its no-gradient forward pass for D produces 512 fake points instead of 256. All G sampling streams and slot assignments are preserved exactly. Larger D batches necessarily change D stream consumption; D slots use a separate seed+80000 stream while unused original D-slot draws preserve the original G-slot sequence. This keeps G’s optimization workload fixed, but changes D sample exposure, gradient noise and compute together. It does not solely test network capacity or functional complexity.','',
        '## Budget accounting','',
        'Equal G updates compares the same number of optimizer steps and generated training points; D512 sees twice as many D points. Equal cumulative D samples compares D512 at t updates to D256 at 2t updates: both see 1,024t total D input points, but D512 has half as many G and D optimizer updates. Neither comparison matches FLOPs or wall time. Curves at equal D samples show only the overlapping budget range.','',
        'Extra 5k/25k/75k checkpoints support exact equal-D-sample endpoint comparisons. Fixed 10,000-sample evaluations use the original seed-specific noise. Coarse metrics every 1k. Coverage uses ≥1% of all generated samples and ≥8% of ideal mode mass; validity radius 0.3. Mode TV penalizes invalid mass and allocation errors. Fine-grid TV uses exact Gaussian mixture probabilities in 0.05-wide cells over [-5.5,5.5]² plus an outside category. Real-sample calibration remains the same.','']
    for accounting in ('equal_g_updates','equal_d_samples'):
        lines += ['## '+accounting.replace('_',' ').capitalize(),'','Mean ± sample SD over five seeds. The budget column is the D256 reference update count; actual updates explicitly show the D512 difference.','',
            '| Modes | Reference budget | Method | Actual updates | Mode TV ↓ | Fine-grid TV ↓ | Valid | Coverage ≥1% | Relative coverage |',
            '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['accounting']!=accounting:continue
            fmt=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}"
            lines.append(f"| {r['modes']} | {r['reference_steps']//1000}k | {r['method']} | {r['actual_steps']//1000}k | {fmt('mode_tv')} | {fmt('spatial_tv')} | {r['valid_fraction']['mean']:.1%} | {r['coverage_1pct']['mean']:.1f}/{r['modes']} | {r['coverage_relative']['mean']:.1f}/{r['modes']} |")
        lines+=['',f'![Curves](curves_{accounting}.png)','',f'![Endpoint metrics](endpoints_{accounting}.png)','']
    lines+=['## Samples and mode mass','','![Mode mass maps](mode_mass_maps.png)','']
    for side in (3,5,7):
        for step in (10000,150000 if side==7 else 50000):
            lines.append(f'- [{side}×{side}, {step//1000}k: all seeds, D256 versus D512](samples_grid{side}_{step//1000}k.png)')
    lines+=['','## Verification and reproduction','',
        'All 425 retained endpoints were recalculated and verified, including 225 new D512 endpoints and 200 reused baseline endpoints. Every new endpoint regenerates bit-for-bit from its checkpoint, and optimizer counters match its update budget. G reference/noise/slot/evaluation RNG states match the corresponding baseline at all 75 mutually retained checkpoints. Old sample hashes match the previous verified report. All training source hashes and configurations verified. Tests include exact original-trainer reproduction at D256 and D-only workload doubling with unchanged G streams.','',
        '`uv run python -m paired_discriminator.grid_d_batch_run` refuses existing result directories. Rebuild with `uv run python -m paired_discriminator.grid_d_batch_report results/grid-d512-comparison-v1`. Full metrics, matched-seed differences and mean/SD summaries for both accounting schemes are retained as CSV/JSON. Checkpoints are kept locally and ignored by Git; source, plots, metrics and small sample arrays are retained for Git.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    import sys
    build(sys.argv[1])
