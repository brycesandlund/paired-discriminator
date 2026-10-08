"""Render the retrospective CIFAR class-frequency audit."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cifar_class_representation import CLASSES, CLASSIFIERS


def report(root):
    root = Path(root)
    rows = json.loads((root / 'metrics.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(rows) == 12
    methods = ['vanilla', 'rsgan', 'paired']
    colors = ['#4477AA', '#EEAA33', '#228855']
    flat = []
    for row in rows:
        c = row['confidence']['0.9']
        flat.append({k: row[k] for k in ['method', 'step', 'seed', 'classifier', 'class_tv', 'covered_classes_1pct']} |
                    {'acceptance_0.9': c['acceptance'], 'augmented_tv_0.9': c['augmented_tv'],
                     'confidence_covered_classes_1pct': c['covered_classes_1pct']})
    with (root / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=flat[0]); writer.writeheader(); writer.writerows(flat)
    for accepted in [False, True]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
        for i, name in enumerate(CLASSIFIERS):
            for j, step in enumerate([10000, 50000]):
                ax = axes[i, j]
                for k, method in enumerate(methods):
                    r = next(r for r in rows if r['classifier'] == name and r['step'] == step and r['method'] == method)
                    mass = r['confidence']['0.9']['accepted_mass'] if accepted else r['class_mass']
                    ax.bar(np.arange(10) + (k-1)*.25, np.array(mass)*100, width=.24, color=colors[k], label=method)
                ax.axhline(10, color='black', ls='--', lw=1, label='training target')
                ax.set_xticks(range(10), CLASSES, rotation=40, ha='right')
                ax.set_title(f'{name} · {step//1000}k updates')
                ax.set_ylabel('% of all generated images'); ax.set_ylim(bottom=0)
        axes[0, 0].legend(ncol=2, fontsize=9)
        fig.suptitle(('Confidence ≥0.9 class mass (no renormalization)' if accepted else 'Predicted class representation') + '\nUnconditional CIFAR-10 · seed 0 · 10,000 fixed samples per checkpoint')
        fig.tight_layout(); fig.savefig(root / ('accepted_mass.png' if accepted else 'class_mass.png'), dpi=160); plt.close(fig)
    lines = ['# Unconditional CIFAR-10: retrospective class representation', '',
        'Existing vanilla, RSGAN and paired checkpoints, seed 0, at 10k and 50k updates. No GAN training or checkpoint changes.', '',
        '## Interpretation', '',
        'At 50k, both classifiers rank paired best on raw class TV, followed by RSGAN and vanilla; at 10k, paired ranks worst. All three methods cover all ten predicted classes at the 1% threshold at 50k; at 10k, automobile is below 1% for every method under both classifiers. This is a late class-balance signal that the FID comparison did not expose, not proof of a general quality advantage or the proposed reference-comparison mechanism.', '',
        'The confidence-filtered result is mixed: at 50k, VGG ranks paired best on augmented TV, while ResNet ranks RSGAN best and paired only slightly ahead of vanilla. Paired has lower confidence acceptance than both baselines under both evaluators. Classifier agreement on generated images at 50k is only 55.82–59.24%, versus 93.93% on real test images. Many gallery images are ambiguous. Thus the agreement on the aggregate raw-TV ranking is encouraging but does not establish accurate semantic proportions. One training seed limits inference.', '',
        '## Protocol', '',
        'Regenerate the original FID evaluation draw: 10,000 samples, CPU noise seed 99174, batches of 128, G in eval mode, rounded to uint8. The same noise is used across models and endpoints. CIFAR-10 training classes are exactly balanced (10% each).', '',
        'Two frozen pretrained classifiers are used as a sensitivity check, not an ensemble: ResNet56 and VGG16-BN from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models), pinned to commit `786c16252c0fc58ee9adac063f8337cc4a7a497a`. Inputs retain native 32×32 resolution; RGB/255 is normalized by mean [0.4914, 0.4822, 0.4465] and std [0.2023, 0.1994, 0.2010], following the author’s [configuration](https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf). Official test images are used only for evaluator diagnostics.', '',
        'Class TV = ½ Σᵢ |qᵢ − 0.1|, where qᵢ is predicted class frequency. Coverage requires ≥1% of all generated samples in a class. Confidence-filtered mass aᵢ counts predictions with max softmax ≥0.9, divided by ALL samples. Augmented TV = ½(Σᵢ |aᵢ − 0.1| + rejected fraction). No accepted-sample renormalization. Threshold sensitivity at 0.5, 0.7, 0.9 and 0.95 is retained in metrics.json.', '',
        '## Evaluator checks', '', '| Classifier | Real-test accuracy | Real-test class TV | Confidence ≥0.9 |', '|---|---:|---:|---:|']
    for name, r in manifest['classifiers'].items():
        lines.append(f"| {name} | {100*r['accuracy']:.2f}% | {r['class_tv']:.4f} | {100*r['confidence']['0.9']['acceptance']:.2f}% |")
    lines += ['', '## Results', '', '| Classifier | Updates | Method | Class TV ↓ | Classes ≥1% | Acceptance ≥0.9 | Augmented TV ↓ |', '|---|---:|---|---:|---:|---:|---:|']
    for r in flat:
        lines.append(f"| {r['classifier']} | {r['step']} | {r['method']} | {r['class_tv']:.4f} | {r['covered_classes_1pct']} | {100*r['acceptance_0.9']:.1f}% | {r['augmented_tv_0.9']:.4f} |")
    lines += ['', '![Class proportions](class_mass.png)', '', '![Accepted class mass](accepted_mass.png)', '',
        '## Limits and inspection', '',
        'One GAN seed; endpoints share a training trajectory. Classifier accuracy on real images does not establish accuracy on generated images. Softmax confidence is uncalibrated and is not proof that an image is recognizable. The two evaluators share a training dataset and may share errors. These metrics do not measure within-class diversity. No evaluator was selected based on the GAN ranking.', '',
        'Per-image probabilities and generated uint8 arrays are retained locally and on Modal (ignored by Git). Manifest records source, checkpoint, generated-image and classifier-weight hashes, GPU and software version, real-test confusion matrices, and classifier agreement. Galleries show the first ten samples per predicted class in fixed draw order, without confidence ranking.', '']
    for method in methods:
        for step in [10000, 50000]:
            key = f'{method}_{step:06d}'
            lines.append(f"- {method}, {step//1000}k: classifier agreement {100*manifest['checkpoints'][key]['classifier_agreement']:.1f}%; " + ', '.join(f'[{name}]({key}_{name}.png)' for name in CLASSIFIERS))
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    import sys
    report(sys.argv[1])
