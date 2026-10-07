"""Held-out image quality and two-classifier audit of deficit CIFAR checkpoints."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .cifar import configure, models, atomic_json
from .cifar_deficit import load_classifier, WEIGHTS
from .cifar_class_representation import predict, summarize, gallery, digest, REPO
from .integrity_eval import extract, extractor, metrics


def evaluate(run_dir, data_root, step, device='cuda', commit=None):
    configure(device)
    run_dir, data_root = Path(run_dir), Path(data_root)
    output = run_dir / f'eval_{step:06d}'
    output.mkdir(exist_ok=True)
    checkpoint = run_dir / f'generator_{step:06d}.pt'
    checkpoint_hash = digest(checkpoint)
    destination = output / 'metrics.json'
    if destination.exists():
        result = json.loads(destination.read_text())
        if result['checkpoint_sha256'] != checkpoint_hash or result['evaluation_sha256'] != digest(__file__):
            raise ValueError('Evaluation provenance mismatch')
        return result
    began = time.perf_counter()
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    g, _ = models(config, ckpt['method'].removesuffix('_deficit'), ckpt['seed'])
    g.load_state_dict(ckpt['generator']); g.to(device).eval()
    rng = torch.Generator().manual_seed(config['eval_seed'] + 1)
    n = config['eval_samples']
    with torch.inference_mode():
        fake = torch.cat([((g(torch.randn(min(128,n-i), config['latent_dim'], generator=rng).to(device)).cpu()+1)*127.5).round().clamp(0,255).byte() for i in range(0,n,128)])
    np.save(output / 'images.npy', fake.numpy())
    cache = data_root / 'integrity-v1'
    real = torch.from_numpy(np.load(cache / 'test_uint8.npy'))
    labels = np.load(cache / 'test_labels.npy')
    result = {'method': ckpt['method'], 'seed': ckpt['seed'], 'step': step,
              'checkpoint_sha256': checkpoint_hash, 'evaluation_sha256': digest(__file__),
              'generated_sha256': hashlib.sha256(fake.numpy().tobytes()).hexdigest(),
              'samples': n, 'reference': 'official CIFAR10 test, all 10000; never used in GAN updates',
              'classifiers': {}, 'real_classifiers': {}, 'classifier_repository': REPO,
              'classifier_weight_hashes': WEIGHTS,
              'reference_manifest': json.loads((cache / 'manifest.json').read_text())}
    predictions = []
    for name in WEIGHTS:
        model = load_classifier(name, data_root.parent / 'class-representation-weights', device)
        real_prob = predict(model, real, device)
        real_stats = summarize(real_prob)
        real_stats['accuracy'] = float((real_prob.argmax(1) == labels).mean())
        if real_stats['accuracy'] <= .90:
            raise ValueError('Classifier calibration failed')
        result['real_classifiers'][name] = real_stats
        prob = predict(model, fake, device)
        np.savez_compressed(output / f'{name}.npz', probabilities=prob)
        result['classifiers'][name] = summarize(prob)
        predictions.append(prob.argmax(1))
        gallery(fake, prob, output / f'{name}.png', f"{ckpt['method']}, {step:,} updates — {name}")
        del model
    result['classifier_agreement'] = float((predictions[0] == predictions[1]).mean())
    ext = extractor(device)
    features = extract(fake.numpy(), ext, device)
    del ext
    heldout = torch.load(cache / 'test_features.pt', map_location='cpu', weights_only=True)
    train = torch.load(cache / 'train_features.pt', map_location='cpu', weights_only=True)
    ids = np.load(cache / 'reference_ids.npz')['train_metrics']
    result['heldout_metrics'] = metrics(features, heldout, device)
    result['train_metrics'] = metrics(features, train[ids], device)
    result['evaluation_seconds'] = time.perf_counter() - began
    (output / 'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(destination, result)
    if commit: commit()
    return result
