"""Collect and report experiment 38 without selecting on confirmation seeds."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .cifar_pair_arch import ARCHITECTURES, SOURCES
from .cifar_tune import digest

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'results/cifar-pair-arch-v1'
LABELS = {'concat':'Original concat', 'concat_wide':'Wider concat', 'shared_difference':'Shared independent scores',
          'global_context':'Global reference context', 'self_attention':'Self-attention control', 'cross_attention':'Cross-attention',
          'symmetric_concat':'Antisymmetric concat (probe)'}


def collect():
    import modal
    v = modal.Volume.from_name('paired-discriminator-cifar-pair-architecture')
    entries = [e for e in v.iterdir('/cifar-pair-arch-v1', recursive=True)
               if Path(e.path).suffix in ('.json','.png','.py')]
    def fetch(e):
        rel = Path(e.path.lstrip('/')).relative_to('cifar-pair-arch-v1')
        dest = R/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b''.join(v.read_file(e.path)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, entries))
    print('Collected',len(entries),'small artifacts',flush=True)


def load(p): return json.loads(p.read_text())


def confirmation_report(rows):
    """Fixed recipes only; do not rank or select on the new seeds."""
    if not (R/'confirmation_decision.json').exists(): return []
    decision=load(R/'confirmation_decision.json')
    names=decision['controls']+[decision['primary_candidate'],decision['targeted_probe']['architecture']]
    chosen={(r['architecture'],r['seed'],r['weights']):r for r in rows if r['step']==decision['endpoint']}
    keys=['train_fid','fid','kid','precision','recall','vgg_tv','resnet_tv']
    lines=['## Frozen fresh-seed confirmation', '',
        'Cross-attention was selected from the six seed-0 arms by train-reference FID. '
        'The following recipes, 50k endpoint, EMA weights and evaluation noise were fixed before seeds 1 and 2 ran. '
        'No checkpoint or weight selection is made on these seeds. Antisymmetric concat is a separate, targeted exploratory probe. '
        'Means below average individual-seed metrics; distributions are never pooled before scoring.', '',
        '| Architecture | Seed | Train FID ↓ | Test FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for name in names:
        rs=[]
        for seed in decision['fresh_confirmation_seeds']:
            r=chosen.get((name,seed,'ema'))
            if r is None:continue
            rs.append(r)
            lines.append('| '+LABELS[name]+f' | {seed} | '+' | '.join(f'{r[k]:.5f}' for k in keys)+' |')
        if len(rs)==2:
            stats={k:dict(mean=float(np.mean([r[k] for r in rs])),sample_sd=float(np.std([r[k] for r in rs],ddof=1))) for k in keys}
            summary.append(dict(architecture=name,seeds=[1,2],weights='ema',step=50000,metrics=stats))
            lines.append('| '+LABELS[name]+' | mean ± SD | '+' | '.join(f"{stats[k]['mean']:.5f} ± {stats[k]['sample_sd']:.5f}" for k in keys)+' |')
    lines+=['', 'Two fresh seeds measure some initialization sensitivity; they are not a precise estimate of the full seed distribution. '
        'Class TV measures classifier label balance, not within-class diversity. Feature recall uses the existing reference/evaluation protocol. '
        'Neither metric alone establishes absence of mode collapse.', '',
        '### Matched changes relative to original concat', '',
        'EMA at 50k; differences are candidate minus original within each training seed. Negative FID/TV and positive recall favor the candidate.', '',
        '| Architecture | Seed | Δ train FID | Δ test FID | Δ recall | Δ VGG TV | Δ ResNet TV |',
        '|---|---:|---:|---:|---:|---:|---:|']
    deltas=[]
    for name in names:
        if name=='concat':continue
        for seed in (1,2):
            r,b=chosen.get((name,seed,'ema')),chosen.get(('concat',seed,'ema'))
            if r is None or b is None:continue
            delta={k:r[k]-b[k] for k in ('train_fid','fid','recall','vgg_tv','resnet_tv')}
            deltas.append(dict(architecture=name,seed=seed,delta=delta))
            lines.append('| '+LABELS[name]+f' | {seed} | '+' | '.join(f'{v:+.5f}' for v in delta.values())+' |')
    lines+=['', '### Raw versus EMA at the same endpoint', '',
        'Descriptive only; the primary weights remain EMA even when raw happens to win.', '',
        '| Architecture | Seed | Raw FID | EMA FID | Raw recall | EMA recall | Raw VGG TV | EMA VGG TV |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in (0,1,2):
            raw,ema=chosen.get((name,seed,'model')),chosen.get((name,seed,'ema'))
            if raw is None or ema is None:continue
            lines.append('| '+LABELS[name]+f' | {seed} | '+' | '.join(f'{r[k]:.5f}' for k in ('fid','recall','vgg_tv') for r in (raw,ema))+' |')
    probe=chosen.get(('symmetric_concat',0,'ema'))
    if probe:
        lines+=['', f"The symmetry probe's seed-0 50k EMA endpoint: train FID {probe['train_fid']:.3f}, "
            f"test FID {probe['fid']:.3f}, recall {probe['recall']:.3f}, VGG TV {probe['vgg_tv']:.3f}. "
            'This targeted follow-up is separate from the original six-arm selection.']
    lines+=['', 'The probe evaluates 0.5 × (D(A,B) − D(B,A)) with the original concat CNN and unchanged parameter count/initialization. '
        'It isolates slot antisymmetry from shared encoders and attention at additional D compute cost. '
        'The copied training loop has explicit parity tests and includes the new source in its resume signature.', '',
        '### Fixed confirmation samples', '', '![Confirmation samples](confirmation_samples.png)', '',
        'All 64 saved samples per run, with no filtering. Each column is one frozen architecture; rows are independent training seeds. '
        'The fixed latent noise is shared across architectures within this phase, but semantic content is not aligned.', '',
        '### Predicted class mass by individual seed', '', '![Confirmation class mass](confirmation_class_mass.png)', '']
    fig,axes=plt.subplots(2,len(names),figsize=(18,9.5),squeeze=False)
    for j,name in enumerate(names):
        for i,seed in enumerate((1,2)):
            ax=axes[i,j];p=R/f'{name}_seed{seed}/eval_050000_ema/samples.png'
            if p.exists():ax.imshow(plt.imread(p))
            else:ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes)
            ax.axis('off');ax.set_title(f'{LABELS[name]}\nseed {seed}',fontsize=10)
    fig.suptitle('Fresh-seed confirmation · fixed 50k EMA · symmetry arm exploratory')
    fig.subplots_adjust(top=.90,bottom=.025,left=.01,right=.99,wspace=.06,hspace=.30)
    fig.savefig(R/'confirmation_samples.png',dpi=150);plt.close(fig)
    classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    for ax,classifier,label in zip(axes,['cifar10_vgg16_bn','cifar10_resnet56'],['VGG','ResNet']):
        data=np.full((2*len(names),10),np.nan);labels=[]
        for j,name in enumerate(names):
            for i,seed in enumerate((1,2)):
                labels.append(f'{LABELS[name]} · seed {seed}')
                r=chosen.get((name,seed,'ema'))
                if r:data[2*j+i]=load(R/r['path'])['classifiers'][classifier]['class_mass']
        im=ax.imshow(data,vmin=0,vmax=.3,cmap='YlOrRd',aspect='auto')
        ax.set(xticks=range(10),xticklabels=classes,yticks=range(len(labels)),yticklabels=labels,title=f'{label} class mass · target .10 each')
        ax.tick_params(axis='x',rotation=45,labelsize=9);ax.tick_params(axis='y',labelsize=8)
        for (i,j),v in np.ndenumerate(data):
            if np.isfinite(v):ax.text(j,i,f'{100*v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>.2 else 'black')
        fig.colorbar(im,ax=ax,fraction=.025,pad=.02,format=PercentFormatter(xmax=1))
    fig.suptitle('Each row is one trained model · values in percent · no pooled-seed scoring')
    fig.tight_layout();fig.savefig(R/'confirmation_class_mass.png',dpi=150);plt.close(fig)
    (R/'confirmation_summary.json').write_text(json.dumps(dict(frozen_decision=decision,individual_seed_metrics=summary,matched_deltas=deltas),indent=2)+'\n')
    return lines


def final_interpretation(rows,statuses):
    """Interpret the completed, predeclared experiment; no new recipe selection."""
    names=['concat','shared_difference','self_attention','cross_attention','symmetric_concat']
    fresh={name:[r for r in rows if r['architecture']==name and r['seed'] in (1,2)
                 and r['step']==50000 and r['weights']=='ema'] for name in names}
    means={name:{k:float(np.mean([r[k] for r in rs])) for k in
                 ('train_fid','fid','kid','precision','recall','vgg_tv','resnet_tv')} for name,rs in fresh.items()}
    minutes={name:float(np.mean([s['training_seconds']/60 for s in statuses
                 if s['folder'] in (f'{name}_seed1',f'{name}_seed2')])) for name in names}
    lines=['## Result', '',
        'Cross-attention improves mean FID and feature recall over original paired, but worsens class balance. '
        'It beats its matched self-attention control on FID in both fresh seeds. Its mean FID is almost identical to '
        'the cheaper independent shared-scoring control. This is evidence of a quality/recall tradeoff, not a demonstrated solution to mode collapse.', '',
        'Fresh seeds 1 and 2, fixed 50k EMA; means of individual-seed scores. Tuned seed 0 is excluded from this table.', '',
        '| Architecture | Train FID ↓ | Test FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ | Training min/seed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m=means[name]
        lines.append(f"| {LABELS[name]} | {m['train_fid']:.2f} | {m['fid']:.2f} | {m['kid']:.5f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['vgg_tv']:.3f} | {m['resnet_tv']:.3f} | {minutes[name]:.2f} |")
    lines+=['',
        'Cross-attention essentially ties original paired on seed 1 (FID 50.27 vs 50.31), but improves substantially '
        'on seed 2 (47.85 vs 53.06). It improves recall in both, while both classifiers report worse class balance. '
        'VGG assigns only 2.1–2.4% to cars and about 20% to frogs in the cross-attention runs, versus 3.1–3.3% and '
        '14.4–15.6% respectively for original paired; the target is 10% per class. These are classifier predictions, not ground-truth generated labels.', '',
        'Independent shared scoring improves FID, precision and recall over original paired in both seeds and has less '
        'FID variation in these two runs. Relative to cross-attention it has higher precision and better class balance, '
        'but lower recall and worse KID. Its mean KID is also slightly worse than original paired, so its FID gain is not '
        'an across-metric win. None of the tested variants improves original paired class TV on either fresh seed.', '',
        'The targeted symmetry probe is inconsistent: it worsens FID and recall on seed 1, improves both on seed 2, '
        'and worsens class TV on both. Its seed-0 probe was also worse than original concat. Enforcing correct slot '
        'antisymmetry alone did not produce a reliable improvement in these runs. Self-attention increases recall but '
        'has worse FID and much worse class balance than original paired.', '',
        'EMA improves FID over raw weights for every 50k endpoint in the confirmation set. The raw/EMA table below '
        'is descriptive; EMA was fixed before these runs, and class TV does not always improve with EMA.', '',
        f"Cross-attention takes {minutes['cross_attention']/minutes['concat']:.2f}× the original paired training time "
        f"and {minutes['cross_attention']/minutes['shared_difference']:.2f}× independent shared scoring on the measured A10 runs. "
        'All deploy the identical one-pass generator, so generator parameter count and inference work are unchanged.', '',
        'The diagnostics confirm a reference-dependent gradient direction in cross-attention, while shared independent '
        'scoring and self-attention have additive logits. Original concat also strongly depends on its reference, so '
        'the result cannot be explained by cross-attention merely starting to use the second image. The useful architectural '
        'change is not isolated to interaction: shared encoding and relative scoring also change the discriminator. '
        'These experiments support retaining independent shared scoring as a strong control and cross-attention as a '
        'recall-oriented variant, rather than claiming that richer comparisons solved class underrepresentation.', '',
        'All fixed sample grids and class-mass plots were inspected. Images remain small and often ambiguous; the grids '
        'do not establish a dramatic visual improvement or within-class coverage. Two fresh seeds and previously used '
        'data references make this an exploratory result, not untouched validation.', '']
    return lines


def build():
    rows=[]; raw_records=[]
    for p in sorted(R.glob('*_seed*/eval_*/metrics.json')):
        r=load(p); folder=p.parent.parent; name,seed=folder.name.rsplit('_seed',1)
        m=r['heldout_metrics']; c=r['classifiers']
        row=dict(architecture=name,seed=int(seed),step=r['step'],weights=r['signature']['weights'],
                 train_fid=r['train_metrics']['frechet_inception_distance'],fid=m['frechet_inception_distance'],
                 kid=m['kernel_inception_distance_mean'],precision=m['precision'],recall=m['recall'],
                 vgg_tv=c['cifar10_vgg16_bn']['class_tv'],resnet_tv=c['cifar10_resnet56']['class_tv'],
                 acceptance=c['cifar10_vgg16_bn']['confidence']['0.9']['acceptance'],path=str(p.relative_to(R)))
        rows.append(row);raw_records.append(r)
    (R/'all_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    primary=[r for r in rows if r['architecture'] in ARCHITECTURES and r['seed']==0 and r['step']==50000 and r['weights']=='ema']
    primary.sort(key=lambda r:r['train_fid'])
    lines=['# Paired CIFAR architecture comparisons (#38)','',
      f'Collected {len(rows)} evaluations. {len(primary)}/6 seed-0 primary endpoints available.', '',
      'Primary comparison: fresh training for 50k updates, EMA weights, train-reference FID for architecture selection. '
      'Raw weights and earlier checkpoints are descriptive. Same generator, BCE, Adam settings, batches, real draws and evaluation noise. '
      'Training compute differs and is measured. Held-out reference scores are exploratory, not an untouched final test.', '',
      '## Seed-0 primary comparison', '',
      '| Architecture | Train FID ↓ | Test FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        lines.append('| '+LABELS[r['architecture']]+' | '+' | '.join(f'{r[k]:.5f}' for k in
                     ['train_fid','fid','kid','precision','recall','vgg_tv','resnet_tv'])+' |')
    lines+=['']+confirmation_report(rows)
    lines+=['', '## Seed-0 training trajectories', '', '![Training trajectories](trajectories.png)', '',
            '## Reference interaction diagnostics', '',
            'Hold fake pixels fixed and change the real reference. Additive unary scoring has the same raw-logit gradient '
            'direction even though BCE can rescale it. A lower direction cosine or a nonzero four-pair interaction residual '
            'shows reference dependence, not necessarily useful comparison or diversity.', '',
            '| Architecture | Step | Direction cosine | Interaction RMS | Logit RMS | Swap error |',
            '|---|---:|---:|---:|---:|---:|']
    for name in LABELS:
        for p in sorted(R.glob(f'{name}_seed*/diagnostics_*.json')):
            d=load(p);step=int(p.stem.rsplit('_',1)[1]);cos=d['gradient_direction_cosine']
            seed=int(p.parent.name.rsplit('_seed',1)[1])
            lines.append(f"| {LABELS[name]} (seed {seed}) | {step:,} | {cos:.5f} | {d['interaction_residual_rms']:.5f} | {d['logit_rms']:.5f} | {d['swap_antisymmetry_error']:.5g} |")
    train_seconds=0.;statuses=[]
    for p in R.glob('*_seed*/status.json'):
        s=load(p);train_seconds+=s['training_seconds'];statuses.append(dict(folder=p.parent.name,**s))
    eval_seconds=sum(r['evaluation_seconds'] for r in raw_records)
    budget=dict(training_seconds=train_seconds,evaluation_seconds=eval_seconds,
                gpu_hours=(train_seconds+eval_seconds)/3600,
                excludes='startup, checkpoint/volume I/O, short verification jobs, untimed diagnostics and coordinator CPU time')
    lines+=['', '## Cost and provenance', '',
      f"Recorded training {train_seconds/3600:.2f} GPU-hours + evaluation {eval_seconds/3600:.2f} = {budget['gpu_hours']:.2f} GPU-hours. "
      'Excludes startup, checkpoint/volume I/O, short verification jobs, untimed diagnostics and coordinator CPU time.', '',
      '| Architecture | D parameters | Completed steps | Training minutes |', '|---|---:|---:|---:|']
    for s in sorted(statuses,key=lambda x:x['folder']):
        p=load(R/s['folder']/'provenance.json')
        lines.append(f"| {s['folder']} | {p['parameters_d']:,} | {s['step']:,} | {s['training_seconds']/60:.2f} |")
    if (R/'confirmation_recovery.json').exists():
        lines+=['', 'Orchestration deviation: Modal preempted the confirmation coordinator; its restart stopped at the duplicate-dispatch guard. '
            'A CPU-only recovery collector rejoined the 11 saved job IDs without launching training. '
            'The original training calls and source signatures were preserved. The GPU budget excludes coordinator CPU time. '
            'Details are saved in confirmation_recovery.json and confirmation_recovery_launch.json.']
    lines+=['', 'Shared CNN weights are initialized identically across the shared-encoder arms. '
      'Self-attention and cross-attention have identical initial parameters; only their source of keys/values differs. '
      'Global conditioning has the same total parameter count. All shared arms have an antisymmetric output. '
      'Independent shared scoring and self-attention remain additive unary controls.', '',
      'Attention runs over 8×8 tokens with four heads. Residual scale starts at 0.1 and is learned. '
      'The wider concatenation control has about 0.7% fewer parameters than the interaction arms. '
      'All methods draw 128 real and fake images for D and another 128 real references and fake images for G each update. '
      'At 50k this is 12.8M real draws per arm. Generator architecture and one-pass inference cost are unchanged.', '',
      '## Fixed samples at the primary endpoint', '',
      '![Primary samples](samples_050000_ema.png)', '',
      'All 64 fixed samples; no filtering. A comparison across architectures uses the same latent noise, '
      'but this does not align generated semantic content.', '']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,label in zip(axes.flat,['fid','vgg_tv','recall','precision'],['Test FID ↓','VGG class TV ↓','Feature recall ↑','Feature precision ↑']):
        for name in ARCHITECTURES:
            rs=sorted((r for r in rows if r['seed']==0 and r['weights']=='ema' and r['architecture']==name),key=lambda r:r['step'])
            if rs:ax.plot([r['step']/1000 for r in rs],[r[key] for r in rs],'-o',label=LABELS[name])
        ax.set(xlabel='G/D updates (thousands)',ylabel=label);ax.grid(alpha=.2)
    if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=8)
    fig.suptitle('Paired CIFAR · seed 0 · EMA · unchanged generator');fig.tight_layout()
    fig.savefig(R/'trajectories.png',dpi=150);plt.close(fig)
    if primary:
        fig,axes=plt.subplots(2,3,figsize=(13,10))
        for ax,name in zip(axes.flat,ARCHITECTURES):
            p=R/f'{name}_seed0/eval_050000_ema/samples.png'
            if p.exists():ax.imshow(plt.imread(p))
            else:ax.text(.5,.5,'Pending',ha='center',va='center')
            ax.axis('off');ax.set_title(LABELS[name],pad=10)
        fig.suptitle('50k updates · seed 0 · EMA · fixed samples',y=.98)
        fig.subplots_adjust(top=.92,bottom=.03,hspace=.15,wspace=.08)
        fig.savefig(R/'samples_050000_ema.png',dpi=160);plt.close(fig)
    baseline=load(ROOT/'results/image-best-checkpoints-v1/summary.json')['cifar']['vanilla']['record']
    verification=dict(evaluations=len(rows),first_batch_replays=all(r['first_batch_replay_exact'] for r in raw_records) if rows else None,
                      reference_matches_baseline=all(r['reference_manifest']==baseline['reference_manifest'] for r in raw_records) if rows else None,
                      classifier_matches_baseline=all(r['classifier_weight_hashes']==baseline['classifier_weight_hashes'] for r in raw_records) if rows else None,
                      evaluation_sources_match=all(r['signature']['source_sha256']==digest(Path(__file__).with_name('cifar_tune.py')) for r in raw_records) if rows else None,
                      noise_matches_protocol=all(r['signature']['noise_seed']==(99374 if r['signature']['seed']==0 else 99474) for r in raw_records) if rows else None,
                      training_sources_match=all(load(p)['source_hashes']=={n:digest(Path(__file__).with_name(n+'.py')) for n in
                                                (SOURCES+('cifar_pair_symmetry',) if load(p)['architecture']=='symmetric_concat' else SOURCES)}
                                                for p in R.glob('*_seed*/provenance.json')),
                      screen_primary_endpoints=len(primary),statuses=statuses)
    provenance=[load(p) for p in R.glob('*_seed*/provenance.json')]
    verification['dataset_identities_match']=len({p['data_sha256'] for p in provenance})==1
    verification['training_protocol_matches']=all(p['batch']==128 and p['lr']==.0002 and p['betas']==[.5,.999] and p['ema_decay']==.999 and p['parameters_g']==1183619 for p in provenance)
    checkpoints={}
    for row,record in zip(rows,raw_records):
        checkpoints.setdefault((row['architecture'],row['seed'],row['step']),set()).add(record['signature']['checkpoint_sha256'])
    verification['raw_ema_checkpoint_identities_match']=all(len(hashes)==1 for hashes in checkpoints.values())
    verification['saved_training_sources_match']=all(digest(p.parent/'training_source.py')==digest(Path(__file__).with_name('cifar_pair_symmetry.py' if load(p)['architecture']=='symmetric_concat' else 'cifar_pair_arch.py')) for p in R.glob('*_seed*/provenance.json'))
    checks=load(R/'verification.json') if (R/'verification.json').exists() else []
    if (R/'confirmation_verification.json').exists():checks+=load(R/'confirmation_verification.json')
    verification['cuda_exact_resume_checks']=checks
    if (R/'confirmation_specs.json').exists():
        specs=load(R/'confirmation_specs.json');decision=load(R/'confirmation_decision.json')
        expected={(name,0,step,weight) for name in ARCHITECTURES for step in (10000,20000,50000) for weight in ('model','ema')}
        expected|={(s['architecture'],s['seed'],step,weight) for s in specs for step in s['eval_steps'] for weight in ('model','ema')}
        observed={(r['architecture'],r['seed'],r['step'],r['weights']) for r in rows}
        verification['expected_evaluations']=len(expected)
        verification['missing_evaluations']=[list(x) for x in sorted(expected-observed)]
        verification['unexpected_evaluations']=[list(x) for x in sorted(observed-expected)]
        verification['duplicate_evaluations']=len(rows)!=len(observed)
        verification['frozen_specs_match']=digest(R/'confirmation_specs.json')==decision['specs_sha256']
        verification['frozen_sources_match']=decision['frozen_sources']=={n:digest(Path(__file__).with_name(n+'.py')) for n in decision['frozen_sources']}
        dispatch=load(R/'confirmation_dispatch.json') if (R/'confirmation_dispatch.json').exists() else None
        verification['dispatch_matches_frozen_specs']=bool(dispatch) and [c['spec'] for c in dispatch['calls']]==specs
        phases={name:load(R/f'{name}_status.json') if (R/f'{name}_status.json').exists() else None for name in ('screen','confirmation')}
        verification['coordinator_statuses']=phases
        expected_folders={f'{n}_seed0' for n in ARCHITECTURES}|{f"{s['architecture']}_seed{s['seed']}" for s in specs}
        complete={s['folder'] for s in statuses if s['phase']=='complete' and s['step']==50000 and s.get('evaluated_step')==50000}
        verification['expected_endpoints']=len(expected_folders)
        verification['missing_complete_endpoints']=sorted(expected_folders-complete)
        positive_checks=[v for k,v in verification.items() if isinstance(v,bool) and k!='duplicate_evaluations']
        verification['complete']=(not (expected-observed or observed-expected or expected_folders-complete)
            and not verification['duplicate_evaluations'] and all(positive_checks)
            and {c['architecture'] for c in checks}==set(ARCHITECTURES)|{'symmetric_concat'}
            and all(c['exact_cuda_resume'] for c in checks)
            and any(c['architecture']=='concat' and c['original_control_equivalent'] for c in checks)
            and all(v and v['phase']=='complete' and not v['errors'] for v in phases.values()))
        lines+=['', '## Verification and phase status', '',
            f"{'Complete' if verification['complete'] else 'In progress'}: {len(rows)}/{len(expected)} evaluations, "
            f"{len(complete & expected_folders)}/{len(expected_folders)} complete endpoints; {len(checks)}/7 CUDA resume checks recorded.", '',
            'First-batch replay, frozen sources/specs, common dataset, classifier/reference identities and noise protocol are recorded in '
            '[verification_report.json](verification_report.json). '+('All required checks passed.' if verification['complete'] else 'Missing endpoints are pending; completion requires all checks.'), '']
        if verification['complete']:lines[2:2]=final_interpretation(rows,statuses)
    (R/'verification_report.json').write_text(json.dumps(verification,indent=2)+'\n')
    (R/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    (R/'REPORT.md').write_text('\n'.join(lines))
    (R/'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(primary=primary,budget=budget,verification={k:v for k,v in verification.items() if k!='statuses'}),indent=2))

if __name__=='__main__':
    import sys
    if '--collect' in sys.argv:collect()
    build()
