"""Retrospective best-observed-checkpoint image comparison, without mixing metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'vanilla': 'Vanilla GAN', 'paired': 'Paired GAN', 'flow': 'Flow matching'}
COLORS = ['#4279ad', '#238b57', '#d78c24']

def read(p):
    return json.loads(p.read_text())

def build(flow_id='image-flow-50k-v1', require_complete=True):
    out = ROOT/'results/image-best-checkpoints-v1'
    out.mkdir(parents=True, exist_ok=True)
    flowroot = ROOT/'results'/flow_id
    mnist, cifar = {}, {}
    for method in NAMES:
        mnist[method] = {}
        candidates = range(1000,50001,1000) if method != 'flow' else range(10000,100001,10000)
        for step in candidates:
            paths = [(ROOT/f'results/mnist-v1/{method}_seed{s}/metrics_{step:06d}.json'
                      if method != 'flow' else flowroot/f'mnist_seed{s}/eval_{step:06d}_midpoint64/metrics.json') for s in range(5)]
            if all(p.exists() for p in paths):
                mnist[method][step] = [read(p) for p in paths]
        folder = ROOT/f'results/cifar10-deficit-v1/{method}_seed0' if method != 'flow' else flowroot/'cifar10_seed0'
        cifar[method] = {int(p.parent.name.split('_')[1]): read(p) for p in sorted(folder.glob('eval_*/metrics.json'))
                         if method != 'flow' or p.parent.name.endswith('midpoint64')}
        assert mnist[method] and cifar[method], method
    if require_complete:
        for dataset, seed in [('mnist',s) for s in range(5)]+[('cifar10',0)]:
            status=read(flowroot/f'{dataset}_seed{seed}/status.json')
            expected = 100000 if dataset == 'cifar10' and (flowroot/'extension_100k.json').exists() else 50000
            assert status['phase']=='complete' and status['step']==expected, status
    def mscore(rows,key): return float(np.mean([r[key] for r in rows]))
    selected_m = {m:min(rows,key=lambda s:mscore(rows[s],'mode_tv')) for m,rows in mnist.items()}
    selected_c = {m:min(rows,key=lambda s:rows[s]['heldout_metrics']['frechet_inception_distance']) for m,rows in cifar.items()}
    selected_aug = {m:min(rows,key=lambda s:mscore(rows[s],'confidence_augmented_tv')) for m,rows in mnist.items()}
    common=set.intersection(*[set(rows) for rows in mnist.values()])
    selected_common={m:min(common,key=lambda s:mscore(rows[s],'mode_tv')) for m,rows in mnist.items()}
    lines=['# Best observed checkpoints: vanilla GAN, paired GAN, flow matching', '',
           ('Final comparison.' if require_complete else 'PROVISIONAL: flow extension is still running; results below use available evaluations.'), '',
           'Unconditional, uniform real sampling, original GAN architectures; no deficit sampling. '
           'MNIST: five training seeds (0–4). CIFAR-10: seed 0 only. Each evaluation uses 10,000 generated images.', '',
           '## Selection rule', '',
           'MNIST selects one common step per method minimizing the mean of the five individual digit TVs. '
           'CIFAR selects minimum held-out-reference FID. Every metric in each primary row comes from that selected step. '
           'No seed selection and no metric-by-metric mixing. Raw flow weights, midpoint64 (128 network evaluations/image).', '',
           'These are retrospective best observed results. The CIFAR test reference is used for checkpoint selection, '
           'so it is not an untouched final test. GAN MNIST evaluations occur every1k through50k; flow every10k. '
           'The shared-interval comparison below controls that search-frequency difference. '
           'GAN snapshots at intermediate1k steps were not retained, but their original metrics and sample grids were; '
           'selected5k/8k results are logged evaluations, not freshly regenerated models.', '',
           '## MNIST — selected by digit balance', '',
           '| Method | Updates | Digit TV ↓ | Digits ≥1% | Acceptance ≥.9 ↑ | Augmented TV ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    keys=['mode_tv','covered_digits','confidence_covered_digits','confidence_accepted_fraction','confidence_augmented_tv']
    summary={'selection':dict(mnist=selected_m,cifar=selected_c,mnist_augmented=selected_aug,mnist_common_intervals=selected_common),
             'mnist':{},'cifar':{},'flow_run':flow_id}
    for m in NAMES:
        s=selected_m[m]; rows=mnist[m][s]
        stats={k:dict(mean=mscore(rows,k),std=float(np.std([r[k] for r in rows],ddof=1)),by_seed=[r[k] for r in rows]) for k in keys}
        summary['mnist'][m]=dict(step=s,stats=stats,records=rows)
        def fmt(k): return f"{stats[k]['mean']:.4f} ± {stats[k]['std']:.4f}"
        lines.append(f"| {NAMES[m]} | {s:,} | {fmt('mode_tv')} | {stats['covered_digits']['mean']:.1f} | {fmt('confidence_accepted_fraction')} | {fmt('confidence_augmented_tv')} |")
    lines+=['', 'Mean ± sample SD across seeds. Digit TV measures mismatch to empirical MNIST training frequencies. '
             'Augmented TV adds rejected (confidence<.9) samples as an extra category. '
             'Confidence is a classifier proxy, not calibrated image quality; neither metric measures within-digit diversity.', '',
             '### Individual training seeds', '', '| Method | Seed | Digit TV | Accepted TV | Acceptance |', '|---|---:|---:|---:|---:|']
    for m in NAMES:
        for seed,r in enumerate(mnist[m][selected_m[m]]):
            lines.append(f"| {NAMES[m]} | {seed} | {r['mode_tv']:.4f} | {r['confidence_augmented_tv']:.4f} | {r['confidence_accepted_fraction']:.4f} |")
    lines+=['', '### Selection sensitivity', '', '| Method | Shared10k-grid best step | Digit TV | Best augmented-TV step | Augmented TV | Digit TV there |', '|---|---:|---:|---:|---:|---:|']
    for m in NAMES:
        s,a=selected_common[m],selected_aug[m]
        lines.append(f"| {NAMES[m]} | {s:,} | {mscore(mnist[m][s],'mode_tv'):.4f} | {a:,} | {mscore(mnist[m][a],'confidence_augmented_tv'):.4f} | {mscore(mnist[m][a],'mode_tv'):.4f} |")
    lines+=['', '## CIFAR-10 — selected by FID', '', '| Method | Updates | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | ResNet TV ↓ | VGG TV ↓ |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in NAMES:
        s=selected_c[m]; r=cifar[m][s]; q=r['heldout_metrics']; summary['cifar'][m]=dict(step=s,record=r)
        lines.append(f"| {NAMES[m]} | {s:,} | {q['frechet_inception_distance']:.2f} | {q['kernel_inception_distance_mean']:.5f} | {q['precision']:.3f} | {q['recall']:.3f} | {r['classifiers']['cifar10_resnet56']['class_tv']:.4f} | {r['classifiers']['cifar10_vgg16_bn']['class_tv']:.4f} |")
    lines+=['', 'One training seed; these differences do not establish a seed-robust ranking. '
             'FID/KID use the same held-out Inception reference; classifier identities are checked below. '
             'Class balance can improve without improved within-class coverage.', '',
             '### CIFAR classifier diagnostics at the FID-selected checkpoints', '',
             '| Method | VGG acceptance | VGG augmented TV | ResNet acceptance | ResNet augmented TV | Evaluator agreement |',
             '|---|---:|---:|---:|---:|---:|']
    for m in NAMES:
        r=cifar[m][selected_c[m]]; a=r['classifiers']['cifar10_vgg16_bn']['confidence']['0.9']; b=r['classifiers']['cifar10_resnet56']['confidence']['0.9']
        lines.append(f"| {NAMES[m]} | {a['acceptance']:.4f} | {a['augmented_tv']:.4f} | {b['acceptance']:.4f} | {b['augmented_tv']:.4f} | {r['classifier_agreement']:.4f} |")
    lines+=['', '## Training curves', '', '![Curves](trajectories.png)', '', '## Fixed seed0 samples at selected steps', '',
             'Original fixed sample grids; seed0 was not chosen for performance. GAN and flow noise spaces differ.', '', '![Samples](comparison_samples.png)', '',
             '## Predicted category mass', '', 'MNIST error bars are training-seed SD. Reported TVs average individual seed scores, '
             'not the averaged category distribution. The seed heatmaps expose biases hidden by pooling.', '',
             '![Class mass](class_mass.png)', '', '![Seed mass](mnist_seed_mass.png)', '', '## Budgets', '',
             'Checkpoint selection is not a compute-matched comparison. All use batch128. '
             'Flow consumes128 real draws/update; vanilla GAN D consumes128; paired GAN consumes128 in D plus128 G references. '
             'Flow performs one velocity-network optimization/update; GAN performs D and G optimizations. '
             'Flow sampling uses128 network evaluations/image versus one generator evaluation for GAN. '
             'Architecture, optimizer and runtime differ.', '', '| Dataset / method | Selected real draws | Selected training updates |', '|---|---:|---:|']
    for dataset,sel in [('MNIST',selected_m),('CIFAR',selected_c)]:
        for m,s in sel.items(): lines.append(f'| {dataset} / {NAMES[m]} | {s*128*(2 if m=="paired" else 1):,} | {s:,} |')
    lines += ['', '### Flow cumulative training budget', '',
              '| Dataset / seed | Final updates | Training seconds | Total real draws |', '|---|---:|---:|---:|']
    budgets=[]
    migrations=[]
    for dataset,seed in [('mnist',s) for s in range(5)]+[('cifar10',0)]:
        folder=flowroot/f'{dataset}_seed{seed}'
        status=read(folder/'status.json')
        budgets.append(dict(dataset=dataset,seed=seed,**status))
        lines.append(f"| {dataset} / {seed} | {status['step']:,} | {status['training_seconds']:.1f} | {status['real_samples']:,} |")
        if (folder/'hardware_migration.json').exists():
            migrations.append(dict(dataset=dataset,seed=seed,**read(folder/'hardware_migration.json')))
    lines += ['', 'Times include original10k training and continuations, exclude sampling/evaluation, and sum GPU training time rather than wall-clock time. '
              'The complete search budget exceeds the selected-checkpoint budget above.', '',
              '### Solver sensitivity at selected flow checkpoints', '',
              '| Dataset (seed0) | Checkpoint | Midpoint steps | Digit TV / FID | Acceptance / recall |', '|---|---:|---:|---:|---:|']
    solver_checks={}
    for dataset,step in [('mnist',selected_m['flow']),('cifar10',selected_c['flow'])]:
        for solver in (64,128):
            path=flowroot/f'{dataset}_seed0/eval_{step:06d}_midpoint{solver}/metrics.json'
            if not path.exists():
                assert not require_complete, f'Missing selected-checkpoint solver check: {path}'
                continue
            r=read(path);assert r['first_batch_float_replay_exact']
            metric=r['mode_tv'] if dataset=='mnist' else r['heldout_metrics']['frechet_inception_distance']
            extra=r['confidence_accepted_fraction'] if dataset=='mnist' else r['heldout_metrics']['recall']
            solver_checks[f'{dataset}_{solver}']=r
            lines.append(f'| {dataset} | {step:,} | {solver} | {metric:.5f} | {extra:.5f} |')
    lines += ['', 'Solver checks use identical fixed noise; primary selection stays midpoint64. These seed0 checks do not replace the five-seed MNIST mean.', '',
              '### Resume provenance', '',
              'Original10k results are preserved. Training source, optimizer settings and data are unchanged. '
              'Where Modal assigned A10 versus A10G, hardware_migration.json records the name transition and a replay '
              'of the original10k first128 float samples with maximum absolute error below.001. '
              'Model, optimizer and explicit RNG state were restored; no cross-GPU bitwise-continuation claim is made.']
    for m in migrations:
        lines.append(f"- {m['dataset']} seed{m['seed']}: {m['previous_gpu']} → {m['current_gpu']}; replay max error {m['pilot_first_batch_max_abs_error']:.8g}.")
    summary.update(budgets=budgets,hardware_migrations=migrations,solver_checks=solver_checks)

    # Verify reference identity and saved flow replay results.
    classifier=read(ROOT/'results/mnist-v1/classifier/report.json')
    count=0
    for s,rows in mnist['flow'].items():
        for r in rows:
            assert r['first_batch_float_replay_exact']
            assert r['evaluation_signature']['classifier_sha256']==classifier['checkpoint_sha256']
            count+=1
    for s,r in cifar['flow'].items():
        assert r['first_batch_float_replay_exact']; count+=1
        for m in ['vanilla','paired']:
            b=cifar[m][selected_c[m]]
            assert b['reference_manifest']==r['reference_manifest']
            assert b['classifier_weight_hashes']==r['classifier_weight_hashes']
    if require_complete:
        assert count == 25 + len(cifar['flow'])
        assert set(mnist['flow']) == set(range(10000,50001,10000))
        assert set(cifar['flow']) == set(range(10000,max(cifar['flow'])+1,10000))
    (out/'verification.json').write_text(json.dumps(dict(primary_flow_evaluations=count,first_batch_replays=count,
        classifier_and_reference_identity=True,all_jobs_complete=require_complete,selected_solver_checks=len(solver_checks)),indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for color,m in zip(COLORS,NAMES):
        x=sorted(mnist[m]);
        for ax,k in zip(axs[:2],['mode_tv','confidence_augmented_tv']):
            y=np.array([mscore(mnist[m][s],k) for s in x]); sd=np.array([np.std([r[k] for r in mnist[m][s]],ddof=1) for s in x])
            ax.plot(np.array(x)/1000,y,color=color,label=NAMES[m]); ax.fill_between(np.array(x)/1000,y-sd,y+sd,color=color,alpha=.12)
        x=sorted(cifar[m]); axs[2].plot(np.array(x)/1000,[cifar[m][s]['heldout_metrics']['frechet_inception_distance'] for s in x],color=color,label=NAMES[m])
    for ax,title in zip(axs,['MNIST digit TV ↓ · mean ± SD','MNIST augmented TV ↓ · mean ± SD','CIFAR held-out FID ↓ · seed0']):
        ax.set(title=title,xlabel='Training updates (thousands)');ax.grid(alpha=.2)
    axs[0].legend();fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for col,m in enumerate(NAMES):
        for row,(dataset,sel) in enumerate([('mnist',selected_m),('cifar10',selected_c)]):
            s=sel[m]
            p=(flowroot/f'{dataset}_seed0/eval_{s:06d}_midpoint64/samples.png' if m=='flow' else
               ROOT/f'results/{"mnist-v1" if dataset=="mnist" else "cifar10-deficit-v1"}/{m}_seed0/samples_{s:06d}.png')
            axs[row,col].imshow(plt.imread(p));axs[row,col].axis('off');axs[row,col].set_title(f'{dataset.upper()} · {NAMES[m]} · {s//1000}k')
    fig.suptitle('Best observed checkpoints · fixed seed0 samples');fig.tight_layout();fig.savefig(out/'comparison_samples.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(17,4.8))
    from .cifar_class_representation import CLASSES
    for i,(color,m) in enumerate(zip(COLORS,NAMES)):
        mass=np.array([r['digit_mass'] for r in mnist[m][selected_m[m]]])*100
        axs[0].bar(np.arange(10)+(i-1)*.25,mass.mean(0),.25,yerr=mass.std(0,ddof=1),label=NAMES[m],color=color,error_kw={'elinewidth':.7})
        for ax,cl in zip(axs[1:],['cifar10_resnet56','cifar10_vgg16_bn']):
            ax.bar(np.arange(10)+(i-1)*.25,np.array(cifar[m][selected_c[m]]['classifiers'][cl]['class_mass'])*100,.25,color=color)
    axs[0].plot(np.arange(10),np.array(classifier['target_digit_mass'])*100,'k--',label='Training target');axs[0].legend(fontsize=8)
    axs[0].set(title='MNIST · mean ± training-seed SD',xticks=np.arange(10),ylabel='% of generated images')
    for ax,title in zip(axs[1:],['CIFAR · ResNet56','CIFAR · VGG16-BN']):
        ax.axhline(10,color='k',linestyle='--');ax.set(title=title,xticks=np.arange(10),xticklabels=CLASSES);ax.tick_params(axis='x',rotation=55)
    fig.tight_layout();fig.savefig(out/'class_mass.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    masses=[np.array([r['digit_mass'] for r in mnist[m][selected_m[m]]])*100 for m in NAMES]
    for ax,m,mass in zip(axs,NAMES,masses):
        im=ax.imshow(mass,vmin=0,vmax=max(a.max() for a in masses),cmap='viridis',aspect='auto')
        ax.set(title=f'{NAMES[m]} · {selected_m[m]//1000}k',xlabel='Predicted digit',ylabel='Training seed',xticks=range(10),yticks=range(5))
        for (y,x),v in np.ndenumerate(mass): ax.text(x,y,f'{v:.1f}',ha='center',va='center',fontsize=7,color='white' if v<9 else 'black')
    fig.colorbar(im,ax=axs,label='% of generated images');fig.savefig(out/'mnist_seed_mass.png',dpi=170);plt.close(fig)
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary['selection'],indent=2))

if __name__=='__main__':
    build()
