"""Sync small CIFAR deficit artifacts and plot complete or partial endpoints."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

METHODS=('vanilla','paired','vanilla_deficit','paired_deficit')
NAMES={'vanilla':'Vanilla','paired':'Paired','vanilla_deficit':'Vanilla + deficit','paired_deficit':'Paired + deficit'}


def sync(root,run_id):
    import modal
    volume=modal.Volume.from_name('paired-discriminator-cifar10-runs')
    for entry in volume.iterdir('/'+run_id,recursive=True):
        path=Path(entry.path)
        if path.suffix not in ('.json','.csv','.png','.py','.toml','.lock'):
            continue
        rel=Path(*path.parts[path.parts.index(run_id)+1:])
        if 'verification' in rel.parts and path.name!='verification.json':
            continue
        if path.name.startswith('samples_') and int(path.stem.split('_')[1])%10000:
            continue
        target=root/rel
        # Endpoints and source are immutable; status and training logs keep changing.
        if target.exists() and path.name not in ('status.json','training.csv'):
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        content=b''.join(volume.read_file(entry.path))
        target.write_bytes(content)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .cifar_class_representation import CLASSES
    records=[]
    for method in METHODS:
        for path in sorted((root/f'{method}_seed0').glob('eval_*/metrics.json')):
            records.append(json.loads(path.read_text()))
    for plot_methods, filename in [(('vanilla_deficit', 'paired_deficit'), 'sampling_weights.png'),
                                   (('vanilla', 'paired'), 'sampling_weights_originals.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
        for col, method in enumerate(plot_methods):
            path = root / f'{method}_seed0' / 'training.csv'
            if not path.exists():
                continue
            with path.open() as handle:
                history = list(csv.DictReader(handle))
            for row, prefix in enumerate(('accepted_ema_', 'next_real_weight_')):
                ax = axes[row, col]
                for k, label in enumerate(CLASSES):
                    ax.plot([float(r['step'])/1000 for r in history],
                            [float(r[prefix+str(k)]) for r in history], label=label)
                ax.axhline(.1, color='black', linestyle=':', linewidth=1)
                ax.set_title(NAMES[method] + (' — accepted mass EMA' if row == 0 else ' — next D real probabilities'))
                ax.set_xlabel('G/D updates (thousands)'); ax.grid(alpha=.2)
        if filename == 'sampling_weights_originals.png':
            for ax in axes[1]:
                ax.set_ylim(.045, .165)
                ax.text(.5, .75, 'All 10 classes overlap at 10%', transform=ax.transAxes,
                        ha='center', fontsize=9)
        axes[0, 0].legend(ncol=2, fontsize=7)
        fig.tight_layout(); fig.savefig(root/filename, dpi=170); plt.close(fig)
    if not records:
        return
    # Raw predicted class representation, matching the earlier retrospective chart.
    endpoints = [step for step in (10000, 50000, 100000)
                 if all(any(r['method'] == method and r['step'] == step for r in records)
                        for method in ('vanilla', 'paired'))]
    if endpoints:
        fig, axes = plt.subplots(2, len(endpoints), figsize=(6 * len(endpoints), 9),
                                 sharey=True, squeeze=False)
        x = np.arange(10)
        for row, classifier in enumerate(('cifar10_resnet56', 'cifar10_vgg16_bn')):
            for col, step in enumerate(endpoints):
                ax = axes[row, col]
                ax.axhline(10, color='black', linestyle='--', linewidth=1,
                           label='training target')
                for method, offset, color in [('vanilla', -.18, '#4477aa'),
                                               ('paired', .18, '#228855')]:
                    result = next(r for r in records if r['method'] == method and r['step'] == step)
                    mass = np.asarray(result['classifiers'][classifier]['class_mass'])
                    ax.bar(x + offset, 100 * mass, width=.36, color=color, label=method)
                ax.set_xticks(x, CLASSES, rotation=45, ha='right')
                ax.set_title(f'{classifier} · {step // 1000}k updates')
                ax.set_ylabel('% of all generated images')
                ax.set_ylim(bottom=0)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle('Predicted class representation — original vanilla and paired\n'
                     'Unconditional CIFAR-10 · seed 0 · 10,000 fixed samples per checkpoint\n'
                     'Raw predicted classes; no confidence filtering', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(root / 'class_representation_originals.png', dpi=170)
        plt.close(fig)
    if all(any(r['method'] == method for r in records) for method in METHODS):
        best = [min((r for r in records if r['method'] == method),
                    key=lambda r: (r['heldout_metrics']['frechet_inception_distance'], r['step']))
                for method in METHODS]
        selection = {'criterion': 'minimum held-out FID among evaluated checkpoints; earliest step breaks ties',
                     'selection_is_retrospective': True,
                     'checkpoints': [{'method': r['method'], 'step': r['step'],
                                      'heldout_fid': r['heldout_metrics']['frechet_inception_distance']}
                                     for r in best]}
        (root / 'best_fid_checkpoints.json').write_text(json.dumps(selection, indent=2) + '\n')
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
        x = np.arange(10)
        for ax, classifier in zip(axes, ('cifar10_resnet56', 'cifar10_vgg16_bn')):
            ax.axhline(10, color='black', linestyle='--', linewidth=1, label='training target')
            for index, (result, color) in enumerate(zip(best, ('#4477aa', '#228855', '#eeaa33', '#aa4499'))):
                mass = np.asarray(result['classifiers'][classifier]['class_mass'])
                label = f"{NAMES[result['method']]} · {result['step']//1000}k · FID {result['heldout_metrics']['frechet_inception_distance']:.2f}"
                ax.bar(x + (index - 1.5) * .2, mass * 100, width=.2, color=color, label=label)
            ax.set_xticks(x, CLASSES, rotation=35, ha='right')
            ax.set_ylabel('% of all generated images')
            ax.set_title(classifier)
            ax.set_ylim(bottom=0)
        axes[0].legend(ncol=2, fontsize=9)
        fig.suptitle('Predicted class representation — best FID checkpoint per method\n'
                     'Unconditional CIFAR-10 · seed 0 · 10,000 fixed samples · no confidence filtering\n'
                     'Retrospective selection from checkpoints evaluated every 10k through 100k', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(root / 'class_representation_best_fid.png', dpi=170)
        plt.close(fig)
    rows=[]
    for r in records:
        row={'method':r['method'],'step':r['step'],'seed':r['seed'],**r['heldout_metrics']}
        for classifier in ('cifar10_resnet56','cifar10_vgg16_bn'):
            m=r['classifiers'][classifier]
            row[classifier+'_class_tv']=m['class_tv']
            for threshold in ('0.5','0.7','0.9','0.95'):
                c=m['confidence'][threshold]
                for key in ('acceptance','augmented_tv','covered_classes_1pct'):
                    row[f'{classifier}_{threshold}_{key}']=c[key]
        rows.append(row)
    with (root/'endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    panels=[('frechet_inception_distance','Held-out FID ↓'),('kernel_inception_distance_mean','Held-out KID ↓'),
            ('cifar10_vgg16_bn_class_tv','VGG16-BN raw class TV ↓'),
            ('cifar10_vgg16_bn_0.9_augmented_tv','VGG16-BN accepted-mass TV (≥90%) ↓'),
            ('cifar10_vgg16_bn_0.9_acceptance','VGG16-BN acceptance (≥90%) ↑'),
            ('cifar10_resnet56_0.9_augmented_tv','ResNet56 accepted-mass TV (≥90%) ↓')]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(key,title) in zip(axes.flat,panels):
        for method in METHODS:
            rr=[r for r in rows if r['method']==method]
            ax.plot([r['step']/1000 for r in rr],[r[key] for r in rr],marker='o',label=NAMES[method])
        ax.set_title(title);ax.set_xlabel('G/D updates (thousands)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Unconditional CIFAR-10 • seed 0 • D-only class deficit sampling')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=170);plt.close(fig)
    sets=[{r['step'] for r in records if r['method']==m} for m in METHODS]
    common=set.intersection(*sets)
    step=max(common) if common else None
    lines=['# CIFAR-10 class deficit pilot','',f'{len(records)}/40 endpoint evaluations available. Seed 0 only.',
           '', '[Protocol](../../docs/CIFAR_DEFICIT_PROTOCOL.md). Original unconditional networks; G batch 128; D receives 128 real + 128 fake images per update. Adaptive arms only change D real sampling.','',
           'ResNet56 controls sampling; VGG16-BN is a separate evaluator. Confidence is not ground-truth image validity. FID/KID/precision/recall use the official held-out test split.','',
           '![Learning curves](learning_curves.png)','', '![Sampling weights](sampling_weights.png)', '', '![Original vanilla and paired class mass](sampling_weights_originals.png)', '', '![Original class representation](class_representation_originals.png)', '', 'Best checkpoints below are selected retrospectively by minimum held-out FID, separately for each method.', '', '![Best-FID class representation](class_representation_best_fid.png)', '']
    if step:
        chosen=[next(r for r in records if r['method']==m and r['step']==step) for m in METHODS]
        fig,axes=plt.subplots(1,2,figsize=(15,4.2))
        for ax,name in zip(axes,('cifar10_resnet56','cifar10_vgg16_bn')):
            matrix=np.array([r['classifiers'][name]['confidence']['0.9']['accepted_mass'] for r in chosen])
            im=ax.imshow(matrix,vmin=0,vmax=max(.15,float(matrix.max())),cmap='YlGnBu',aspect='auto')
            ax.set_xticks(range(10),CLASSES,rotation=40,ha='right');ax.set_yticks(range(4),[NAMES[m] for m in METHODS])
            ax.set_title(name+' • ≥90% confidence')
            for i in range(4):
                for j in range(10):
                    ax.text(j,i,f'{matrix[i,j]*100:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>im.norm.vmax*.6 else 'black')
            fig.colorbar(im,ax=ax,fraction=.025)
        fig.suptitle(f'Accepted class mass (% of ALL generated images), {step:,} updates • target 10% each')
        fig.tight_layout();fig.savefig(root/'class_mass.png',dpi=170);plt.close(fig)
        lines += [f'## Latest common endpoint: {step:,} updates','',
                  '| Method | Held-out FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG raw class TV ↓ | VGG accepted TV ↓ | VGG acceptance ↑ |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in chosen:
            q=r['heldout_metrics'];c=r['classifiers']['cifar10_vgg16_bn'];a=c['confidence']['0.9']
            lines.append(f"| {NAMES[r['method']]} | {q['frechet_inception_distance']:.2f} | {q['kernel_inception_distance_mean']:.4f} | {q['precision']:.3f} | {q['recall']:.3f} | {c['class_tv']:.3f} | {a['augmented_tv']:.3f} | {a['acceptance']:.1%} |")
        lines += ['', '![Class mass](class_mass.png)','', 'Accepted TV includes rejected mass as an extra category. Coverage of ten predicted classes does not establish within-class diversity.','']
        for r in chosen:
            lines += [f"### {NAMES[r['method']]}",'',f"![Fixed samples]({r['method']}_seed0/samples_{step:06d}.png)",'']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/cifar10-deficit-v1');p.add_argument('--sync',action='store_true');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    if a.sync: sync(root,root.name)
    report(root)

if __name__=='__main__':main()
