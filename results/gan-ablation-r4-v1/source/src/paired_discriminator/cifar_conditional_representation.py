"""Retrospective classification of frozen CBN/projection CIFAR checkpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .cifar_integrity import configure, models, atomic_json

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
REPO = 'chenyaofo/pytorch-cifar-models:786c16252c0fc58ee9adac063f8337cc4a7a497a'
CLASSIFIERS = ['cifar10_resnet56', 'cifar10_vgg16_bn']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(prob):
    prob = np.asarray(prob)
    assert prob.ndim == 2 and prob.shape[1] == 10 and np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(1), 1, atol=1e-5)
    pred, conf = prob.argmax(1), prob.max(1)
    mass = np.bincount(pred, minlength=10) / len(pred)
    result = {'n': len(pred), 'class_mass': mass.tolist(), 'class_tv': float(abs(mass - .1).sum() / 2),
              'covered_classes_1pct': int((mass >= .01).sum()), 'confidence': {}}
    for threshold in [.5, .7, .9, .95]:
        accepted = conf >= threshold
        a = np.bincount(pred[accepted], minlength=10) / len(pred)
        reject = 1 - accepted.mean()
        result['confidence'][str(threshold)] = {'accepted_mass': a.tolist(), 'acceptance': float(accepted.mean()),
            'covered_classes_1pct': int((a >= .01).sum()),
            'augmented_tv': float((abs(a - .1).sum() + reject) / 2)}
    return result


def conditional_summary(prob, labels):
    result = summarize(prob)
    labels = np.asarray(labels)
    assert len(labels) == len(prob) and set(labels) == set(range(10))
    pred = prob.argmax(1)
    matrix = np.bincount(labels * 10 + pred, minlength=100).reshape(10, 10)
    result['requested_predicted_counts'] = matrix.tolist()
    result['adherence'] = float((pred == labels).mean())
    result['per_class_adherence'] = (matrix.diagonal() / matrix.sum(1)).tolist()
    for threshold, metrics in result['confidence'].items():
        accepted = prob.max(1) >= float(threshold)
        metrics['correct_fraction_all_samples'] = float(((pred == labels) & accepted).mean())
    return result


@torch.inference_mode()
def predict(model, images, device='cuda'):
    mean = torch.tensor([.4914, .4822, .4465], device=device)[None, :, None, None]
    std = torch.tensor([.2023, .1994, .2010], device=device)[None, :, None, None]
    return torch.cat([model((b.to(device).float() / 255 - mean) / std).softmax(1).cpu()
                      for b in images.split(256)]).numpy()


def gallery(images, prob, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred = prob.argmax(1)
    fig, axes = plt.subplots(10, 10, figsize=(10, 11))
    for c in range(10):
        ids = np.flatnonzero(pred == c)[:10]
        for j, ax in enumerate(axes[c]):
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(ids):
                ax.imshow(images[ids[j]].permute(1, 2, 0).numpy())
            if j == 0:
                ax.set_ylabel(CLASSES[c], fontsize=8)
    fig.suptitle(title + '\nFirst 10 per predicted class in fixed draw order; no confidence selection', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(run_root, data_root, output):
    from torchvision.datasets import CIFAR10
    import shutil
    configure('cuda')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)
    test = CIFAR10(str(data_root), train=False, download=True)
    real = torch.from_numpy(test.data.transpose(0, 3, 1, 2).copy())
    labels = np.asarray(test.targets)
    np.save(output / 'test_labels.npy', labels)
    manifest = {'classifier_repository': REPO, 'classes': CLASSES, 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(), 'source_sha256': digest(__file__),
                'test_images_sha256': hashlib.sha256(real.numpy().tobytes()).hexdigest(),
                'normalization_mean': [.4914, .4822, .4465], 'normalization_std': [.2023, .1994, .2010],
                'classifiers': {}, 'checkpoints': {}}
    classifiers = {}
    for name in CLASSIFIERS:
        model = torch.hub.load(REPO, name, pretrained=False, trust_repo=True)
        weight_path = data_root.parent / 'class-representation-weights' / f'{name}.pt'
        expected = {'cifar10_resnet56': '187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b', 'cifar10_vgg16_bn': '6ee7ea24b52cfbbe9751608a81d5a7b2f5bac4e8f7d19e420030072ca257ed97'}
        assert digest(weight_path) == expected[name]
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        model.cuda().eval()
        classifiers[name] = model
        shutil.copy2(weight_path, output / f'{name}.pt')
        prob = predict(model, real)
        np.savez_compressed(output / f'real_{name}.npz', probabilities=prob)
        metrics = summarize(prob)
        metrics['accuracy'] = float((prob.argmax(1) == labels).mean())
        metrics['confusion'] = np.bincount(labels * 10 + prob.argmax(1), minlength=100).reshape(10, 10).tolist()
        metrics['weights_sha256'] = digest(output / f'{name}.pt')
        assert metrics['accuracy'] > .90, metrics
        manifest['classifiers'][name] = metrics
        print(name, 'real test accuracy', metrics['accuracy'], flush=True)
    rows = []
    for method in ['vanilla', 'rsgan', 'paired']:
        for step in range(10000, 100001, 10000):
            path = Path(run_root) / f'{method}_seed0' / f'generator_{step:06d}.pt'
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            assert ckpt['method'] == method and ckpt['seed'] == 0 and ckpt['step'] == step
            config = ckpt['config']
            assert config['width'] == 64 and config['batch_size'] == 128
            assert 'd_width' not in config and 'd_batch_size' not in config
            g, _ = models(config, method, 0)
            g.load_state_dict(ckpt['generator']); g.cuda().eval()
            rng = torch.Generator().manual_seed(config['eval_seed'] + 1)
            with torch.inference_mode():
                fake = torch.cat([((g(torch.randn(min(128, config['eval_samples'] - i), config['latent_dim'], generator=rng).cuda(), torch.arange(i, min(i+128, config['eval_samples']), device='cuda') % 10).cpu() + 1) * 127.5).round().clamp(0, 255).byte()
                                  for i in range(0, config['eval_samples'], 128)])
            requested = np.arange(config['eval_samples']) % 10
            key = f'{method}_{step:06d}'
            previous = json.loads((Path(run_root) / f'{method}_seed0' / f'integrity_{step:06d}.json').read_text())
            generated_hash = hashlib.sha256(fake.numpy().tobytes()).hexdigest()
            assert generated_hash == previous['generated_uint8_sha256'], 'Original evaluation samples must match exactly'
            np.save(output / f'{key}_images.npy', fake.numpy())
            manifest['checkpoints'][key] = {'path': str(path), 'sha256': digest(path), 'config': config,
                'generated_sha256': generated_hash, 'matches_integrity_samples': True, 'heldout_fid': previous['heldout_metrics']['frechet_inception_distance']}
            predictions = {}
            for name, model in classifiers.items():
                prob = predict(model, fake); predictions[name] = prob.argmax(1)
                np.savez_compressed(output / f'{key}_{name}.npz', probabilities=prob)
                row = {'method': method, 'step': step, 'seed': 0, 'classifier': name, **conditional_summary(prob, requested)}
                rows.append(row)
                if step in [10000, 50000, 100000]:
                    gallery(fake, prob, output / f'{key}_{name}.png', f'{method}, {step:,} updates — {name}')
                print(key, name, row['class_tv'], row['adherence'], flush=True)
            manifest['checkpoints'][key]['classifier_agreement'] = float((predictions[CLASSIFIERS[0]] == predictions[CLASSIFIERS[1]]).mean())
    shutil.copy2(__file__, output / 'evaluation_source.py')
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'metrics.json', rows)
    return rows
