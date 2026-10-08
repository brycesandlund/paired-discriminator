"""Report class balance and requested-class adherence over frozen training trajectories."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cifar_conditional_representation import CLASSIFIERS, CLASSES, conditional_summary, digest

METHODS = ['vanilla', 'rsgan', 'paired']
COLORS = ['#4477AA', '#EEAA33', '#228855']


def report(root, baseline):
    root, baseline = Path(root), Path(baseline)
    rows = json.loads((root/'metrics.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert len(rows) == 60
    assert len({(r['method'], r['step'], r['classifier']) for r in rows}) == 60
    for row in rows:
        key = f"{row['method']}_{row['step']:06d}"
        p = np.load(root/f"{key}_{row['classifier']}.npz")['probabilities']
        assert conditional_summary(p, np.arange(len(p)) % 10) == {k:v for k,v in row.items() if k not in ['method','step','seed','classifier']}
        checkpoint = baseline/f"{row['method']}_seed0"/f"generator_{row['step']:06d}.pt"
        assert digest(checkpoint) == manifest['checkpoints'][key]['sha256']
        assert manifest['checkpoints'][key]['matches_integrity_samples']
    for key, entry in manifest['checkpoints'].items():
        assert hashlib.sha256(np.load(root/f'{key}_images.npy').tobytes()).hexdigest() == entry['generated_sha256']
    assert digest(root/'evaluation_source.py') == manifest['source_sha256']
    real = [np.load(root/f'real_{c}.npz')['probabilities'].argmax(1) for c in CLASSIFIERS]
    real_agreement = float((real[0] == real[1]).mean())
    for c in CLASSIFIERS:
        assert digest(root/f'{c}.pt') == manifest['classifiers'][c]['weights_sha256']
    verification = {'records_verified':60, 'checkpoints_verified':30, 'matches_original_integrity_images':True,
                    'real_classifier_agreement':real_agreement}
    (root/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    flat=[]
    for r in rows:
        key=f"{r['method']}_{r['step']:06d}"
        flat.append({k:r[k] for k in ['method','step','seed','classifier','class_tv','adherence','covered_classes_1pct']} |
                    {'acceptance':r['confidence']['0.9']['acceptance'], 'confident_correct':r['confidence']['0.9']['correct_fraction_all_samples'],
                     'augmented_tv':r['confidence']['0.9']['augmented_tv'],
                     'classifier_agreement':manifest['checkpoints'][key]['classifier_agreement'],
                     'heldout_fid':manifest['checkpoints'][key]['heldout_fid']})
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=flat[0]);writer.writeheader();writer.writerows(flat)
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    for i,c in enumerate(CLASSIFIERS):
        for j,(metric,title) in enumerate([('class_tv','Class TV ↓'),('adherence','Requested-class adherence ↑'),('confident_correct','Correct AND confidence ≥0.9 ↑')]):
            ax=axes[i,j]
            for method,color in zip(METHODS,COLORS):
                rr=sorted([r for r in flat if r['method']==method and r['classifier']==c], key=lambda r:r['step'])
                ax.plot([r['step']/1000 for r in rr],[r[metric] for r in rr],marker='o',label=method,color=color)
            ax.set_title(c+'\n'+title);ax.set_xlabel('Training updates (thousands)');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            if j>0:ax.set_ylim(0,1)
    axes[0,0].legend();fig.suptitle('Conditional CIFAR-10 · CBN + projection · original D · seed 0 · 10,000 images/checkpoint')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    for step in [10000,50000,100000]:
        fig,axes=plt.subplots(2,3,figsize=(14,9))
        for i,c in enumerate(CLASSIFIERS):
            for j,method in enumerate(METHODS):
                r=next(r for r in rows if r['method']==method and r['classifier']==c and r['step']==step)
                ax=axes[i,j];mat=np.array(r['requested_predicted_counts'])/1000
                im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues')
                ax.set_xticks(range(10),CLASSES,rotation=60,ha='right',fontsize=7);ax.set_yticks(range(10),CLASSES,fontsize=7)
                ax.set_title(f"{method} · {c}\nadherence {r['adherence']:.1%}",fontsize=10)
                ax.set_xlabel('Predicted');ax.set_ylabel('Requested')
        fig.suptitle(f'Requested → predicted class, {step//1000}k updates · row proportions, scale 0–1')
        fig.tight_layout();fig.savefig(root/f'confusion_{step:06d}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for i,c in enumerate(CLASSIFIERS):
        for j,step in enumerate([50000,100000]):
            ax=axes[i,j]
            for k,(method,color) in enumerate(zip(METHODS,COLORS)):
                r=next(r for r in rows if r['method']==method and r['classifier']==c and r['step']==step)
                ax.bar(np.arange(10)+(k-1)*.25,np.array(r['class_mass'])*100,width=.24,color=color,label=method)
            ax.axhline(10,color='black',ls='--',lw=1);ax.set_xticks(range(10),CLASSES,rotation=40,ha='right');ax.set_title(f'{c} · {step//1000}k');ax.set_ylabel('% of all generated images')
    axes[0,0].legend();fig.suptitle('Predicted class balance · balanced requested labels · conditional CIFAR-10')
    fig.tight_layout();fig.savefig(root/'class_mass.png',dpi=160);plt.close(fig)
    lines=['# Conditional CIFAR-10: class representation and adherence','',
           'Frozen conditional BatchNorm G + projection D checkpoints from the integrity study, seed 0: vanilla, RSGAN, original paired. Every 10k through 100k; no GAN retraining. D width 64 and batch 128 throughout; the wider-D and doubled-D-batch arms are excluded.','',
           '## Findings','',
           'Paired has the lowest requested-class adherence at every evaluated checkpoint under both classifiers. At 100k it reaches 64.41% / 67.43% (ResNet / VGG), versus vanilla 68.89% / 72.07% and RSGAN 68.72% / 71.60%. Confident-correct fractions at 100k also favor the baselines. Longer training substantially improves all methods, and paired continues improving through 100k.', '',
           'At 50k and 100k, vanilla has better marginal class TV than paired under both classifiers. At 100k RSGAN also beats paired under both. Paired has a small VGG-only TV win at 80k, but there is no consistent class-balance advantage across the trajectory. All methods have ten predicted classes above 1% by 50k and at 100k. Thus the unconditional late class-balance ranking does not carry over cleanly to this conditional setup.', '',
           'At 100k the classifiers agree on 74.30% of vanilla, 74.40% of RSGAN and 71.67% of paired images, versus 93.93% on real test images. This is better agreement than the unconditional audit but still a substantial limitation. Birds and dogs are among the difficult requested classes for all methods; galleries and confusion matrices show ambiguity and class confusion. This result does not test the wider paired D or larger D batch.', '',
           '## Protocol','',
           '10,000 generated images per checkpoint, with 1,000 requests per class (`arange(10000) % 10`), CPU noise seed 99174, batches of 128, G in eval mode, rounded uint8. Generated-image hashes match the earlier integrity evaluation exactly at all 30 checkpoints. The model factory is from `cifar_integrity.py`.','',
           'Reuse the two frozen [CIFAR classifiers](https://github.com/chenyaofo/pytorch-cifar-models) from the unconditional audit: ResNet56 and VGG16-BN, pinned commit `786c16252c0fc58ee9adac063f8337cc4a7a497a`, verified weight hashes. Normalization and native 32×32 images unchanged. Neither evaluator provides a GAN training signal.','',
           'Class TV = ½Σ|predicted class frequency − 0.1|. Coverage means ≥1% of ALL images assigned to a class. Adherence is the fraction with predicted class equal to requested class; its per-class breakdown and requested/predicted count matrix are retained. Confident-correct is the fraction of ALL images both correctly classified for the request and max softmax ≥0.9. Accepted class mass is likewise divided by ALL images. Augmented TV = ½(Σ|accepted class mass − 0.1| + rejected fraction). Thresholds 0.5/0.7/0.9/0.95 are retained.','',
           'Balanced requests prescribe the intended class mix; marginal balance alone cannot establish adherence (even systematically swapped labels can be balanced). Neither class balance nor adherence measures within-class diversity.','',
           '## Evaluator diagnostics','', '| Evaluator | Real-test accuracy | Real-test class TV |','|---|---:|---:|']
    for c in CLASSIFIERS:
        r=manifest['classifiers'][c];lines.append(f"| {c} | {r['accuracy']:.2%} | {r['class_tv']:.4f} |")
    lines += ['', f'Real-test classifier agreement: {real_agreement:.2%}. Softmax confidence is uncalibrated. Real-image accuracy does not establish generated-image accuracy.', '', '## Endpoint results', '',
              '| Updates | Evaluator | Method | Class TV ↓ | Adherence ↑ | Confident-correct ↑ | Acceptance ≥0.9 |', '|---:|---|---|---:|---:|---:|---:|']
    for r in flat:
        if r['step'] in [10000,50000,100000]:
            lines.append(f"| {r['step']} | {r['classifier']} | {r['method']} | {r['class_tv']:.4f} | {r['adherence']:.1%} | {r['confident_correct']:.1%} | {r['acceptance']:.1%} |")
    lines+=['','![Learning curves](learning_curves.png)','','![Class proportions](class_mass.png)','','## Confusion matrices','']
    for step in [10000,50000,100000]:lines.append(f'- [{step//1000}k requested-versus-predicted classes](confusion_{step:06d}.png)')
    lines+=['','## Galleries and agreement','','Galleries take the first ten samples per predicted class in fixed draw order, without confidence selection. Existing integrity class grids show fixed noise across requested classes.','']
    for method in METHODS:
        for step in [10000,50000,100000]:
            key=f'{method}_{step:06d}'
            lines.append(f"- {method}, {step//1000}k, classifier agreement {manifest['checkpoints'][key]['classifier_agreement']:.1%}: "+', '.join(f'[{c}]({key}_{c}.png)' for c in CLASSIFIERS)+f'; [requested-class grid](../cifar10-integrity-v1/{method}_seed0/class_grid_{step:06d}.png).')
    lines+=['','## Limits and reproducibility','','One GAN training seed; checkpoints and fixed evaluation draws are correlated. Two classifiers share training data and can share biases. Confidence is not a validity oracle; interpret differences alongside galleries and existing FID/KID. No causal claim about how D uses its pair follows.','',
            'Run `uv run --extra cifar modal run --detach modal_conditional_representation.py::main` with the existing CIFAR volumes (classifier weights already cached). Download `/cifar10-conditional-representation-v1` from the run volume, then run `uv run python -m paired_discriminator.cifar_conditional_representation_report results/cifar10-conditional-representation-v1 results/cifar10-integrity-v1`. Report generation verifies all 60 records, 30 checkpoint hashes, original image hashes, classifier weights and source snapshot. Large arrays/checkpoints remain local and on Modal, ignored by Git.','']
    (root/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    import sys
    report(*sys.argv[1:])
