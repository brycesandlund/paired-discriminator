"""Summarize the image flow pilot against the original 10k GAN endpoints."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def build(run_id='image-flow-v1'):
    root = ROOT/'results'/run_id
    launch = read(root/'launch.json')
    assert launch['verification']['cuda_exact_resume']
    assert launch['verification']['sampling_does_not_change_training']
    old_classifier = read(ROOT/'results/mnist-v1/classifier/report.json')
    assert old_classifier['checkpoint_sha256'] == launch['verification']['classifier_sha256']
    statuses, mnist, cifar, checks = [], [], {}, {}
    for dataset, seed in [('mnist', s) for s in range(5)] + [('cifar10', 0)]:
        folder = root/f'{dataset}_seed{seed}'
        status = read(folder/'status.json')
        assert status['phase'] == 'complete' and status['step'] == 10000
        statuses.append(dict(dataset=dataset, seed=seed, **status))
        provenance = read(folder/'provenance.json')
        assert hashlib.sha256((folder/'training_source.py').read_bytes()).hexdigest() == provenance['source_sha256']
        for solver in [64] + ([128] if seed == 0 else []):
            evaluation = folder/f'eval_010000_midpoint{solver}'
            r = read(evaluation/'metrics.json')
            assert r['first_batch_float_replay_exact'] and r['samples'] == 10000
            assert r['dataset'] == dataset and r['seed'] == seed and r['weights'] == 'raw'
            assert hashlib.sha256((evaluation/'evaluation_source.py').read_bytes()).hexdigest() == r['signature']['evaluation_source_sha256']
            if dataset == 'mnist':
                assert r['evaluation_signature']['classifier_sha256'] == launch['verification']['classifier_sha256']
                assert r['evaluation_signature']['target_digit_mass'] == old_classifier['target_digit_mass']
                if solver == 64:
                    mnist.append(r)
            else:
                cifar[solver] = r
            if seed == 0:
                checks[f'{dataset}_{solver}'] = r
    baseline = read(ROOT/'results/mnist-v1/summary.json')['10000']
    keys = ['mode_tv', 'covered_digits', 'confidence_covered_digits',
            'confidence_accepted_fraction', 'confidence_augmented_tv']
    flow = {key: dict(mean=float(np.mean([r[key] for r in mnist])),
                     std=float(np.std([r[key] for r in mnist], ddof=1)),
                     by_seed=[r[key] for r in mnist]) for key in keys}
    mnist_summary = {**baseline, 'flow_matching': flow}
    gan_cifar = {method: read(ROOT/f'results/cifar10-deficit-v1/{method}_seed0/eval_010000/metrics.json')
                 for method in ('vanilla', 'paired')}
    for r in gan_cifar.values():
        assert r['reference_manifest'] == cifar[64]['reference_manifest']
        assert r['classifier_weight_hashes'] == cifar[64]['classifier_weight_hashes']
    summary = dict(mnist=mnist_summary, cifar={**gan_cifar, 'flow_matching': cifar[64]},
                   solver_checks=checks, budgets=statuses)
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Unconditional flow matching on MNIST and CIFAR (#35)', '',
             'Completed initial10k-update pilot: MNIST seeds0–4 and CIFAR seed0. '
             'Uniform real sampling, no labels or deficit. Independent-pair straight-line velocity MSE. '
             'Raw weights and midpoint64 are primary; no checkpoint or seed selection. '
             'Baseline GANs are existing original10k endpoints; no GAN retraining.', '',
             '## MNIST', '', 'Five-seed mean ± sample SD; 10,000 samples per seed.', '',
             '| Method | Digit TV ↓ | Digits ≥1% | Accepted digits ≥1% | Acceptance ↑ | Augmented TV ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for method, r in mnist_summary.items():
        def fmt(key):
            return f"{r[key]['mean']:.4f} ± {r[key]['std']:.4f}"
        lines.append(f"| {method} | {fmt('mode_tv')} | {r['covered_digits']['mean']:.1f} | "
                     f"{r['confidence_covered_digits']['mean']:.1f} | {fmt('confidence_accepted_fraction')} | "
                     f"{fmt('confidence_augmented_tv')} |")
    lines += ['', 'The original frozen classifier and empirical training digit proportions are unchanged. '
              'Confidence ≥.9 is an uncalibrated proxy. Accepted mass uses all generated images as denominator. '
              'Digit coverage requires ≥1% per digit, not merely a nonzero count. No within-digit diversity conclusion.', '',
              '### Flow individual seeds', '', '| Seed | Digit TV | Coverage | Acceptance | Augmented TV |',
              '|---|---:|---:|---:|---:|']
    for r in mnist:
        lines.append(f"| {r['seed']} | {r['mode_tv']:.4f} | {r['covered_digits']} | "
                     f"{r['confidence_accepted_fraction']:.4f} | {r['confidence_augmented_tv']:.4f} |")
    lines += ['', '## CIFAR', '', 'Seed0 only, 10,000 generated images. Official held-out test reference for image metrics.', '',
              '| Method | Held-out FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | ResNet class TV ↓ | VGG class TV ↓ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method, r in summary['cifar'].items():
        m = r['heldout_metrics']
        lines.append(f"| {method} | {m['frechet_inception_distance']:.2f} | "
                     f"{m['kernel_inception_distance_mean']:.5f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{r['classifiers']['cifar10_resnet56']['class_tv']:.4f} | "
                     f"{r['classifiers']['cifar10_vgg16_bn']['class_tv']:.4f} |")
    lines += ['', '### Classifier diagnostics', '',
              '| Method | Evaluator agreement | ResNet acceptance ≥.9 | VGG acceptance ≥.9 | ResNet accepted TV ↓ | VGG accepted TV ↓ |',
              '|---|---:|---:|---:|---:|---:|']
    for method, r in summary['cifar'].items():
        a = r['classifiers']['cifar10_resnet56']['confidence']['0.9']
        b = r['classifiers']['cifar10_vgg16_bn']['confidence']['0.9']
        lines.append(f"| {method} | {r['classifier_agreement']:.4f} | {a['acceptance']:.4f} | "
                     f"{b['acceptance']:.4f} | {a['augmented_tv']:.4f} | {b['augmented_tv']:.4f} |")
    lines += ['', 'Training-reference FID/KID and full classifier confidence diagnostics remain in summary.json. '
              'Use matching reference sets when comparing with older pilot tables. CIFAR has one seed; '
              'classifier predictions on generated images can be wrong, and class balance is not within-class diversity.', '',
              '## Solver sensitivity (seed0)', '',
              '| Dataset | Midpoint steps | NFE/image | Main metric | Confidence acceptance / recall | Sampling seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for dataset in ('mnist', 'cifar10'):
        for solver in (64, 128):
            r = checks[f'{dataset}_{solver}']
            metric = f"digit TV {r['mode_tv']:.4f}" if dataset == 'mnist' else f"FID {r['heldout_metrics']['frechet_inception_distance']:.2f}"
            extra = r['confidence_accepted_fraction'] if dataset == 'mnist' else r['heldout_metrics']['recall']
            lines.append(f"| {dataset} | {solver} | {2*solver} | {metric} | {extra:.4f} | {r['sampling_seconds']:.1f} |")
    lines += ['', 'Same fixed noise within each solver check. All samples are integrated without clipping; '
              'endpoint conversion to uint8 rounds/clamps exactly as GAN evaluation does. '
              'Out-of-range pixel fractions and first-batch float replay checks are retained.', '',
              '## Compute and architecture', '',
              'Three-scale convolutional U-Net, widths32/64/128, GroupNorm/SiLU residual blocks and '
              'sinusoidal time embedding. 1,168,929 parameters for MNIST, 1,170,083 for CIFAR. '
              'No attention or pretrained features. Adam .0002, betas(.9,.999), warmup500, batch128. '
              'Same full training splits and normalization as GANs. No augmentation.', '',
              '| Dataset / seed | Training seconds | Real draws | Evaluation seconds (all solvers) |',
              '|---|---:|---:|---:|']
    for s in statuses:
        lines.append(f"| {s['dataset']} / {s['seed']} | {s['training_seconds']:.1f} | {s['real_samples']:,} | "
                     f"{sum(e['evaluation_seconds'] for e in s['evaluations']):.1f} |")
    lines += ['', 'Flow matches GAN D real draws at10k (1.28M), but paired GAN additionally uses G references. '
              'Architectures and optimizer updates differ. Sampling uses128 velocity-network evaluations per image '
              'instead of one GAN generator pass. These are matched-data/update endpoints, not equal-compute results. '
              'Full evaluation is at10k rather than every1k; every1k preview uses only16 midpoint steps.', '',
              '## Samples', '', 'Fixed seed0, raw weights, 10k updates; no selection for quality.', '',
              '![Samples](comparison_samples.png)', '',
              '## Verification', '', 'Four local tests passed. CUDA20 versus10+10 updates match model, EMA, optimizer, '
              'explicit RNG states and losses for both datasets; intervening sampling is neutral. '
              'All six jobs complete, eight final evaluation records, source hashes and first-batch replays verified. '
              'Frozen classifier/reference identities match the GAN evaluations. No extra training or commits.', '']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    paths = [ROOT/f'results/mnist-v1/{m}_seed0/samples_010000.png' for m in ('vanilla', 'paired')]
    paths += [root/'mnist_seed0/eval_010000_midpoint64/samples.png']
    paths += [ROOT/f'results/cifar10-deficit-v1/{m}_seed0/samples_010000.png' for m in ('vanilla', 'paired')]
    paths += [root/'cifar10_seed0/eval_010000_midpoint64/samples.png']
    for ax, path, title in zip(axes.flat, paths,
                               ['MNIST vanilla', 'MNIST paired', 'MNIST flow', 'CIFAR vanilla', 'CIFAR paired', 'CIFAR flow']):
        ax.imshow(plt.imread(path))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('10k updates · fixed seed0 samples · raw weights')
    fig.tight_layout()
    fig.savefig(root/'comparison_samples.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    digit_target = old_classifier['target_digit_mass']
    for method_index, method in enumerate(('vanilla', 'paired', 'flow_matching')):
        if method == 'flow_matching':
            masses = np.array([r['digit_mass'] for r in mnist])
        else:
            masses = np.array([read(ROOT/f'results/mnist-v1/{method}_seed{s}/metrics_010000.json')['digit_mass'] for s in range(5)])
        axes[0].bar(np.arange(10)+(method_index-1)*.25, masses.mean(0)*100, .25,
                    yerr=masses.std(0, ddof=1)*100, label=method, error_kw={'elinewidth': .6})
    axes[0].plot(np.arange(10), np.array(digit_target)*100, 'k--', linewidth=1, label='training target')
    axes[0].set(title='MNIST · five-seed mean ± SD', xticks=np.arange(10), ylabel='% of all generated images')
    axes[0].legend(fontsize=8)
    from .cifar_class_representation import CLASSES
    for ax, classifier in zip(axes[1:], ('cifar10_resnet56', 'cifar10_vgg16_bn')):
        for method_index, (method, r) in enumerate(summary['cifar'].items()):
            ax.bar(np.arange(10)+(method_index-1)*.25, np.array(r['classifiers'][classifier]['class_mass'])*100, .25, label=method)
        ax.axhline(10, color='k', linestyle='--', linewidth=1)
        ax.set(title=f'CIFAR · {classifier} · seed0', xticks=np.arange(10), xticklabels=CLASSES)
        ax.tick_params(axis='x', labelrotation=55)
    fig.suptitle('Predicted category mass · 10k updates · raw models · 10k samples')
    fig.tight_layout()
    fig.savefig(root/'class_mass.png', dpi=160)
    plt.close(fig)
    lines += ['## Predicted category mass', '', '![Category mass](class_mass.png)', '']
    lines += ['## Interpretation', '',
              'At10k, flow retains all ten MNIST digits across five seeds. Its raw digit TV is between '
              'the original vanilla and paired means; confidence acceptance and confidence-augmented TV '
              'are better than both. This early checkpoint does not yet test the late imbalance observed '
              'in vanilla at50k. On CIFAR seed0, flow improves10k FID/KID, feature recall and raw class '
              'balance versus both original GANs, but precision is below vanilla and classifier agreement '
              'is only about50%. These mixed metrics and one seed do not establish general superiority '
              'or preservation of within-class diversity. This is an untuned image-flow pilot.', '',
              'A result-export helper initially failed on epoch-zero file timestamps in the Modal mount. '
              'The archive helper was repaired and collection rerun; training and evaluation were unchanged.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'report_source.py').write_bytes(Path(__file__).read_bytes())
    (root/'verification.json').write_text(json.dumps(dict(jobs_complete=6, evaluations_complete=8,
        endpoint_first_batch_replays=8, source_hashes=True, classifier_and_reference_matches=True,
        cuda_exact_resume=True, sampling_neutral=True), indent=2)+'\n')
    print('\n'.join(lines[:35]))


if __name__ == '__main__':
    build()
