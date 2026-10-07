"""Report both class-matched checkpoints and contextual unconditional results."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paired_discriminator.cifar_classmatched import METHODS

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def report(root, unconditional):
    root, unconditional = Path(root), Path(unconditional)
    records = []
    metas = []
    sensitivity = []
    for method in METHODS:
        folder = root / f'{method}_seed0'
        meta = json.loads((folder/'run.json').read_text())
        metas.append(meta)
        assert hashlib.sha256((folder/'training_source.py').read_bytes()).hexdigest() == meta['source_sha256']
        status = json.loads((folder/'status.json').read_text())
        assert status['step'] == status['target_steps'] == 50000
        with (folder/'training.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 500 and int(rows[-1]['step']) == 50000
        assert all(np.isfinite(float(r[k])) for r in rows for k in ('d_loss','g_loss'))
        for step in (10000,50000):
            r = json.loads((folder/f'metrics_{step:06d}.json').read_text())
            assert r['step'] == step and r['seed'] == 0 and r['method'] == method
            assert all(np.isfinite(r[k]) for k in ('frechet_inception_distance','kernel_inception_distance_mean','precision','recall'))
            prior = json.loads((unconditional/f'{method}_seed0'/f'metrics_{step:06d}.json').read_text())
            for k in ('samples_generated','samples_real','reference','features','torch_fidelity'):
                assert r[k] == prior[k], k
            records.append({'experiment':'class-matched', **r})
            records.append({'experiment':'unconditional', **prior})
            grid = np.asarray(Image.open(folder/f'class_grid_{step:06d}.png').convert('RGB'), dtype=np.float32) / 255
            assert grid.shape == (342,274,3)
            tiles = np.stack([np.stack([grid[2+y*34:34+y*34,2+z*34:34+z*34] for z in range(8)]) for y in range(10)])
            label_mae = float(np.mean([np.abs(tiles[a]-tiles[b]).mean() for a in range(10) for b in range(a+1,10)]))
            noise_mae = float(np.mean([np.abs(tiles[:,a]-tiles[:,b]).mean() for a in range(8) for b in range(a+1,8)]))
            sensitivity.append({'method':method,'step':step,'label_change_pixel_mae':label_mae,'noise_change_pixel_mae':noise_mae})
    for key in ('config','source_sha256','dataset_sha256','labels_sha256','torch','device','gpu','parameters_g'):
        assert all(m[key] == metas[0][key] for m in metas), key
    assert metas[0]['dataset_sha256'] == json.loads((unconditional/'vanilla_seed0/run.json').read_text())['dataset_sha256']
    with (root/'comparison.csv').open('w',newline='') as f:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (root/'label_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for step in (10000,50000):
        for kind, name, size in [('samples','samples',(13.5,4.8)),('class_grid','classes',(14,6.5))]:
            fig, axes = plt.subplots(1,3,figsize=size)
            for ax, method in zip(axes,METHODS):
                img = plt.imread(root/f'{method}_seed0'/f'{kind}_{step:06d}.png')
                ax.imshow(img,interpolation='nearest')
                ax.set_title(method)
                if kind == 'class_grid':
                    ax.set_yticks((np.arange(10)+.5)*(img.shape[0]-2)/10+1,CLASSES)
                    ax.set_xticks([])
                else:
                    ax.axis('off')
            title = 'rows = requested class; columns = same latent vector' if kind == 'class_grid' else 'identical fixed latent draws and requested labels'
            fig.suptitle(f'Class-matched CIFAR-10 · {step:,} updates · {title}')
            fig.tight_layout()
            fig.savefig(root/f'{name}_{step:06d}.png',dpi=150)
            plt.close(fig)
    lines = ['# Class-matched CIFAR-10: seed 0', '',
        'All three generators receive a one-hot class label. Real references match the requested class for RSGAN and paired; vanilla sees the same class-balanced image batches. Discriminators receive images only, without labels or auxiliary classification losses.', '',
        'Both 10k and 50k checkpoints come from a single trajectory per method. Each evaluation uses 10,000 generated images (1,000 requested per class) and the same fixed real reference and metrics as before. Lower FID/KID and higher precision/recall are better. These remain FID-10k measurements.', '',
        '| Updates | Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |', '|---:|---|---:|---:|---:|---:|']
    for step in (10000,50000):
        for method in METHODS:
            r = next(r for r in records if r['experiment']=='class-matched' and r['step']==step and r['method']==method)
            lines.append(f"| {step:,} | {method} | {r['frechet_inception_distance']:.2f} | {1000*r['kernel_inception_distance_mean']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    lines += ['', '## Context: unconditional → class-matched at 50k', '',
        'This changes generator conditioning/initialization and sampling as well as pair selection; it does not isolate matching alone. Same seed number does not make this a statistically replicated result.', '',
        '| Method | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        a = next(r for r in records if r['experiment']=='unconditional' and r['step']==50000 and r['method']==method)
        b = next(r for r in records if r['experiment']=='class-matched' and r['step']==50000 and r['method']==method)
        lines.append(f"| {method} | {a['frechet_inception_distance']:.2f} → {b['frechet_inception_distance']:.2f} | {1000*a['kernel_inception_distance_mean']:.2f} → {1000*b['kernel_inception_distance_mean']:.2f} | {a['precision']:.3f} → {b['precision']:.3f} | {a['recall']:.3f} → {b['recall']:.3f} |")
    lines += ['', '## Interpretation limits', '',
        'Matching is by requested label; G may ignore that label. Marginal metrics do not measure semantic label adherence. The class grids below repeat identical latent vectors down each column while changing the requested class. Differences across rows show label sensitivity, but only correct class-specific content would demonstrate adherence. One training seed cannot establish a reliable ranking.', '',
        '[Protocol](CIFAR10_CLASSMATCHED.md) · [All metrics CSV](comparison.csv)', '']
    lines += ['## Label sensitivity in the fixed grids', '',
        'Mean absolute RGB-pixel difference on a [0,1] scale, measured from the saved grids. Changing labels holds noise fixed; changing noise holds the label fixed. Only eight fixed latent vectors are used. This diagnoses sensitivity, not semantic correctness or conditional accuracy.', '',
        '| Updates | Method | Change label | Change noise |', '|---:|---|---:|---:|']
    for r in sorted(sensitivity,key=lambda r:(r['step'],METHODS.index(r['method']))):
        lines.append(f"| {r['step']:,} | {r['method']} | {r['label_change_pixel_mae']:.4f} | {r['noise_change_pixel_mae']:.4f} |")
    lines += ['', 'A generator that ignores y and samples the real marginal makes the randomized real/fake pair distribution symmetric, so this image-only discriminator has no way to penalize label ignorance at that solution. See the protocol for the density argument.', '']
    for step in (10000,50000):
        lines += [f'## {step:,} updates', '', f'![Samples](samples_{step:06d}.png)', '', f'![Requested classes](classes_{step:06d}.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    repo = Path(__file__).parents[2]
    for name in ('CIFAR10.md','CIFAR10_CLASSMATCHED.md'):
        shutil.copy2(repo/'docs'/name,root/name)
    shutil.copy2(__file__,root/'report_source.py')
    (root/'verification.json').write_text(json.dumps({'methods':list(METHODS),'seed':0,'all_runs_completed_50000':True,'endpoint_evaluations':6,'matched_config_images_labels_and_source':True,'same_reference_and_metric_protocol_as_unconditional':True,'finite_losses_and_metrics':True},indent=2)+'\n')
    print('\n'.join(lines[:17]))


if __name__ == '__main__':
    import sys
    report(sys.argv[1],sys.argv[2])
