"""Frozen-recipe confirmation report for experiment37; no confirmation selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cifar_tune_report import ROOT,build
from .cifar_class_representation import CLASSES

R=ROOT/'results/cifar-tune-v1'
ARMS=[('Vanilla EMA','vanilla','control',10000,'ema'),
      ('Paired EMA','paired','control',20000,'ema'),
      ('Flow raw (primary)','flow','control',30000,'model'),
      ('Flow low LR + EMA (secondary)','flow','lr5e5',30000,'ema')]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(method,cfg,step,weight,seed,group='confirm_'):
    folder=R/f'{group}{method}_{cfg}_seed{seed}'/f'eval_{step:06d}_{weight}'
    return read(folder/'metrics.json'),folder

def values(r):
    q=r['heldout_metrics']
    return dict(fid=q['frechet_inception_distance'],kid=q['kernel_inception_distance_mean'],precision=q['precision'],
                recall=q['recall'],vgg_tv=r['classifiers']['cifar10_vgg16_bn']['class_tv'],
                resnet_tv=r['classifiers']['cifar10_resnet56']['class_tv'],
                acceptance=r['classifiers']['cifar10_vgg16_bn']['confidence']['0.9']['acceptance'])

def finalize():
    status=read(R/'confirmation_status.json');assert status['phase']=='complete' and not status['errors']
    confirmation=read(R/'confirmation.json');bases=read(R/'confirmation_bases.json')
    assert len(confirmation['records'])==8 and not confirmation['errors']
    assert len(bases['records'])==6 and not bases['errors']
    recipes=read(R/'confirmation_recipes.json')
    for name,m,cfg,step,w in ARMS:
        recipe=recipes[m]
        assert recipe['branch_steps']==step
        if 'secondary' in name:assert recipe['secondary_config']==cfg and recipe['secondary_weights']==w
        else:assert recipe.get('primary_config',recipe['config']['name'])==cfg and recipe['primary_weights']==w
    build();appendix=(R/'REPORT.md').read_text();(R/'ALL_CANDIDATES.md').write_text(appendix)
    selected={};summary={}
    for name,m,cfg,step,w in ARMS:
        rs=[record(m,cfg,step,w,s)[0] for s in (1,2)];selected[name]=rs
        arr=[values(r) for r in rs]
        summary[name]={k:dict(mean=float(np.mean([a[k] for a in arr])),std=float(np.std([a[k] for a in arr],ddof=1)),
                           seeds=[a[k] for a in arr]) for k in arr[0]}
    lines=['# CIFAR optimization: frozen recipes on new seeds (#37)','',
      'Completed optimizer/EMA screening, refinement and confirmation. Seed0 selected recipes; independent training seeds1,2 '
      'test those fixed recipes with new evaluation noise99274. No checkpoint, seed or weight selection on confirmation results. '
      'Unconditional, uniform sampling; unchanged architectures and adversarial/flow objectives.','',
      '## Findings','',
      'Weight averaging is the clearest reproducible improvement from this search. The frozen secondary flow recipe '
      '(quarter learning rate plus EMA) reaches FID 42.78 and 42.81 on the two new seeds. '
      'The primary raw-flow recipe is less reliable: 49.50 and 43.74, versus 42.11 on the tuning seed. '
      'We retain both frozen choices rather than changing the primary recipe after seeing confirmation results.','',
      'Paired GAN with EMA beats vanilla GAN with EMA on both FID and class TV in both new seeds: '
      'mean FID 49.03 versus 51.38 and VGG class TV 0.160 versus 0.238. Its mean feature recall is slightly lower '
      '(0.343 versus 0.352), so this supports better category balance without establishing better coverage in every sense.','',
      'Vanilla also exposes a limitation of the frozen schedule: its extra 10k updates worsen FID in both seeds. '
      'The earlier 50k EMA bases average FID 49.39, compared with 51.38 after the branch. '
      'We keep the prescribed endpoint in the primary comparison, but this limits any claim of a large paired-versus-vanilla FID advantage. '
      'The paired and flow EMA recipes improve over their corresponding earlier EMA bases in both seeds.','',
      'The learning-rate changes do not show a convincing independent benefit. For flow, the same-checkpoint '
      'control EMA diagnostic reaches 42.57 and 42.86, essentially matching the quarter-rate EMA recipe. '
      'For both GANs the original optimizer remained strongest in screening. Additional flow training and EMA helped; '
      'the seed-0 raw-flow improvement did not reproduce consistently.','',
      '## Confirmation — fixed choices, seeds1 and2','',
      '| Recipe | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG class TV ↓ | ResNet class TV ↓ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in selected:
        a=summary[name]
        fmt=lambda k: f"{a[k]['mean']:.5f} ± {a[k]['std']:.5f}" if k=='kid' else f"{a[k]['mean']:.4f} ± {a[k]['std']:.4f}"
        lines.append('| '+name+' | '+' | '.join(fmt(k) for k in ['fid','kid','precision','recall','vgg_tv','resnet_tv'])+' |')
    lines+=['', 'Mean ± sample SD of two training seeds; this small confirmation is not a strong significance test. '
      'Feature recall and predicted class balance are incomplete coverage measures and do not prove within-class diversity. '
      'All reference images and classifier identities match earlier reports. Shared data-reference history means this is exploratory, not an untouched test-set claim.', '',
      '### Individual seeds', '', '| Recipe | Seed | FID | KID | Precision | Recall | VGG TV | VGG acceptance |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,rs in selected.items():
        for seed,r in zip((1,2),rs):
            v=values(r);lines.append(f"| {name} | {seed} | "+' | '.join(f'{v[k]:.4f}' for k in ['fid','kid','precision','recall','vgg_tv','acceptance'])+' |')
    lines+=['', '### EMA versus raw at the same checkpoint', '',
      '| Method / LR | Seed | Raw FID | EMA FID | Raw VGG TV | EMA VGG TV |', '|---|---:|---:|---:|---:|---:|']
    for _,m,cfg,step,_ in ARMS:
        for seed in (1,2):
            raw=values(record(m,cfg,step,'model',seed)[0]);ema=values(record(m,cfg,step,'ema',seed)[0])
            lines.append(f"| {m} / {cfg} | {seed} | {raw['fid']:.3f} | {ema['fid']:.3f} | {raw['vgg_tv']:.4f} | {ema['vgg_tv']:.4f} |")
    lines+=['', '### Change from each fresh base', '',
      'The base and branch use the same evaluation noise. This comparison includes extra training and an Adam reset; '
      'it does not isolate either intervention. Base raw and EMA scores are both shown to distinguish weight averaging from further training.', '',
      '| Recipe | Seed | Base raw FID | Base EMA FID | Final recipe FID | Base raw VGG TV | Final recipe VGG TV |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,m,cfg,step,w in ARMS:
        base_step=80000 if m=='flow' else 50000
        for seed in (1,2):
            raw=values(record(m,'control',base_step,'model',seed,group='base_')[0])
            ema=values(record(m,'control',base_step,'ema',seed,group='base_')[0])
            final=values(record(m,cfg,step,w,seed)[0])
            lines.append(f"| {name} | {seed} | {raw['fid']:.3f} | {ema['fid']:.3f} | {final['fid']:.3f} | {raw['vgg_tv']:.4f} | {final['vgg_tv']:.4f} |")
    lines+=['', '## What was tested', '',
      '- Six GAN optimizer settings per method: original Adam2e-4, half both learning rates, half D only, half G only, cosine decay, lazy gradient penalty.',
      '- Four flow settings: Adam2e-4,1e-4,5e-5, cosine decay; saved flow EMA at50k/80k/100k.',
      '- Screen10k branch updates on seed0; refine each winner and control through30k. Rank by training-reference FID, inspect diversity diagnostics.',
      '- Freeze primary vanilla control/EMA at50k+10k; paired control/EMA at50k+20k; flow control/raw at80k+30k. Freeze quarter-LR flow/EMA at80k+30k as secondary tradeoff.',
      '- Train fresh bases on seeds1,2, branch identically, evaluate fixed choices. All branches reset Adam and explicit sampling RNG, including controls.', '',
      'EMA decay.999 averages model weights and floating BatchNorm buffers; integer buffers copied. '
      'No architectural feature sweep was performed: this bounded search prioritized confirming optimizer/EMA evidence. '
      'R1 gamma1 was applied every16D updates with interval scaling; paired penalized all paired-input coordinates. '
      'Original GAN controls matched the original implementation in local tests. Fresh flow retains500-step warmup.', '',
      '## Seed0 discovery versus the previous baseline', '',
      '| Method | Previous raw best FID (#36) | Selected tuning FID | Selected tuning VGG TV |', '|---|---:|---:|---:|']
    old=read(ROOT/'results/image-best-checkpoints-v1/summary.json')
    for name,m,cfg,step,w in ARMS:
        new=values(record(m,cfg,step,w,0,group='')[0]);prior=old['cifar']['flow' if m=='flow' else m]['record']['heldout_metrics']['frechet_inception_distance']
        lines.append(f"| {name} | {prior:.3f} | {new['fid']:.3f} | {new['vgg_tv']:.4f} |")
    lines+=['', 'This table is discovery-only: unequal training duration and selection history. '
      'In particular flow received30k updates after an80k warm start with optimizer reset; it is not a pure learning-rate improvement. '
      'The GAN learning-rate and tested gradient-penalty changes did not beat the original-setting EMA controls.', '',
      '## Samples', '', 'All64 fixed samples from each confirmation seed. No filtering by confidence or appearance.', '',
      '![Confirmation samples](confirmation_samples.png)', '', '## Predicted category representation', '',
      'Bars average the two seeds, with training-seed SD; all TV scores above average individual seed TVs, not pooled distributions.', '',
      '![Class mass](confirmation_class_mass.png)', '', '## Budget and verification', '']
    train_seconds=0.;eval_seconds=0.;evals=[];provenances=[]
    for folder in R.iterdir():
        if folder.is_dir() and (folder/'status.json').exists() and (folder/'provenance.json').exists():
            r=read(folder/'status.json')
            assert r['phase']=='complete' and r['step']==r['target'],(folder.name,r)
            train_seconds+=r['training_seconds'];provenances.append(read(folder/'provenance.json'))
    baseline=old['cifar']['vanilla']['record']
    source=sha(ROOT/'src/paired_discriminator/cifar_tune.py')
    for p in R.rglob('metrics.json'):
        r=read(p)
        if 'signature' not in r:continue
        assert r['first_batch_replay_exact']
        assert r['signature']['source_sha256']==source
        assert r['reference_manifest']==baseline['reference_manifest']
        assert r['classifier_weight_hashes']==baseline['classifier_weight_hashes']
        if p.parts[-3].startswith(('confirm_','base_')):assert r['signature']['noise_seed']==99274
        else:assert r['signature']['noise_seed']==99174
        eval_seconds+=r['evaluation_seconds'];evals.append(str(p.relative_to(R)))
    assert len(evals)==79,len(evals) #51 discovery/refinement +12base +16confirmation
    assert len(provenances)==30,len(provenances) #16 discovery +6 fresh bases +8branches
    assert all(p['source_hashes']['cifar_tune']==source for p in provenances)
    for module in ('cifar','image_flow'):
        expected=sha(ROOT/f'src/paired_discriminator/{module}.py')
        assert all(p['source_hashes'][module]==expected for p in provenances)
    assert all(p['data_sha256']==provenances[0]['data_sha256'] for p in provenances)
    budget=dict(training_seconds=train_seconds,evaluation_seconds=eval_seconds,total_gpu_hours=(train_seconds+eval_seconds)/3600,
                excludes='container startup, checkpoint/commit I/O, and tiny verification runs; no failed training attempts observed')
    lines += [f"Recorded training {train_seconds/3600:.2f} GPU-hours + evaluation {eval_seconds/3600:.2f} = {budget['total_gpu_hours']:.2f} GPU-hours. "
       'Excludes container startup, checkpoint/commit I/O and tiny verification runs. Parent checkpoints reused from earlier experiments are not charged again.', '',
       'All79 evaluations have exact first-batch replay flags and matching source/classifier/reference identity; '
       'all30 training jobs have matching trainer and dataset provenance. Three CUDA split-run checks and five local tests passed. '
       'Weights and data remain in Modal volumes; source, metrics and plots are saved here.', '',
       'No compute-matching claim: GANs use one D and one G optimization/update; flow one velocity optimization. '
       'Paired also consumes real references in G updates. EMA adds no deployment network passes. '
       'GAN sampling remains one generator forward; flow remains64 midpoint steps,128 velocity evaluations/image. '
       'The tuning work changes neither architecture nor inference cost.', '',
       '[All candidates](ALL_CANDIDATES.md) · [Machine-readable summary](confirmation_summary.json)', '']
    lines+=['### Cost of reproducing each fixed recipe', '',
      'Per-seed totals include the fresh base plus its selected branch. Training minutes exclude evaluations. '
      'Real-image draws count draws with replacement, not distinct images.', '',
      '| Recipe | Updates | Real draws | Training minutes, seeds 1 / 2 | Sampling seconds per 10k, seeds 1 / 2 |',
      '|---|---:|---:|---:|---:|']
    for name,m,cfg,step,w in ARMS:
        base_step=80000 if m=='flow' else 50000
        minutes=[];sample_seconds=[]
        for seed in (1,2):
            b=read(R/f'base_{m}_control_seed{seed}'/'status.json')
            c=read(R/f'confirm_{m}_{cfg}_seed{seed}'/'status.json')
            minutes.append((b['training_seconds']+c['training_seconds'])/60)
            sample_seconds.append(record(m,cfg,step,w,seed)[0]['sampling_seconds'])
        draws=(base_step+step)*128*(2 if m=='paired' else 1)
        lines.append(f"| {name} | {base_step+step:,} | {draws:,} | {minutes[0]:.2f} / {minutes[1]:.2f} | {sample_seconds[0]:.2f} / {sample_seconds[1]:.2f} |")
    fig,axes=plt.subplots(2,4,figsize=(16,9.6))
    for col,(name,m,cfg,step,w) in enumerate(ARMS):
        for row,seed in enumerate((1,2)):
            _,folder=record(m,cfg,step,w,seed)
            axes[row,col].imshow(plt.imread(folder/'samples.png'));axes[row,col].axis('off')
            axes[row,col].set_title(f'{name}\nseed {seed}',fontsize=11,pad=12)
    fig.suptitle('CIFAR confirmation · frozen recipes · fixed samples',fontsize=16,y=.98)
    fig.subplots_adjust(top=.91,bottom=.03,left=.015,right=.985,hspace=.30,wspace=.05)
    fig.savefig(R/'confirmation_samples.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,cl in zip(axes,['cifar10_resnet56','cifar10_vgg16_bn']):
        for i,(name,rs) in enumerate(selected.items()):
            a=np.array([r['classifiers'][cl]['class_mass'] for r in rs])*100
            ax.bar(np.arange(10)+(i-1.5)*.2,a.mean(0),.2,yerr=a.std(0,ddof=1),label=name,error_kw={'elinewidth':.7})
        ax.axhline(10,color='black',linestyle='--',linewidth=1)
        ax.set(title=cl,xticks=range(10),xticklabels=CLASSES,ylabel='% of all generated images');ax.tick_params(axis='x',rotation=45)
    axes[0].legend(fontsize=8);fig.suptitle('Seeds 1, 2 · mean ± training-seed SD');fig.tight_layout();fig.savefig(R/'confirmation_class_mass.png',dpi=170);plt.close(fig)
    (R/'confirmation_summary.json').write_text(json.dumps(dict(recipes=recipes,metrics=summary,budget=budget,records=selected),indent=2)+'\n')
    (R/'final_verification.json').write_text(json.dumps(dict(evaluations=len(evals),training_jobs=len(provenances),replays=True,
       classifier_reference_identity=True,source_data_identity=True,training_endpoints_complete=True,
       confirmation_recipes_frozen=True,confirmation_noise=99274),indent=2)+'\n')
    (R/'REPORT.md').write_text('\n'.join(lines));(R/'final_report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(metrics=summary,budget=budget),indent=2))

if __name__=='__main__':finalize()
