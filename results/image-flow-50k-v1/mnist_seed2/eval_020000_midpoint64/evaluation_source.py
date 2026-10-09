"""Existing image-protocol metrics applied to fixed-noise flow samples."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .image_flow import SHAPES, make_model, sample, to_uint8
from .mnist import atomic_json, configure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview(model, dataset, config, step, output, device='cuda'):
    from torchvision.utils import save_image
    noise = torch.randn(64, *SHAPES[dataset], generator=torch.Generator().manual_seed(config['eval_seed']))
    points = sample(model, noise.to(device), 16)
    save_image(to_uint8(points).float()/255, Path(output)/f'preview16_{step:06d}.png', nrow=8)


def evaluate(run_dir, dataset_root, step, solver_steps, device='cuda', commit=None):
    from torchvision.utils import save_image
    configure(device)
    run_dir, dataset_root = Path(run_dir), Path(dataset_root)
    checkpoint = run_dir / f'velocity_{step:06d}.pt'
    output = run_dir / f'eval_{step:06d}_midpoint{solver_steps}'
    output.mkdir(exist_ok=True)
    dest = output / 'metrics.json'
    signature = dict(checkpoint_sha256=sha(checkpoint), evaluation_source_sha256=sha(__file__),
                     training_source_sha256=sha(Path(__file__).with_name('image_flow.py')),
                     solver_steps=solver_steps, weights='raw')
    if dest.exists():
        result = json.loads(dest.read_text())
        assert result['signature'] == signature
        return result
    began = time.perf_counter()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config, dataset, seed = (ck['signature'][k] for k in ('config', 'dataset', 'seed'))
    assert ck['step'] == step
    assert ck['signature']['source_sha256'] == signature['training_source_sha256']
    model = make_model(dataset, config, seed).to(device)
    model.load_state_dict(ck['model'])
    rng = torch.Generator().manual_seed(config['eval_seed'])
    images, outside, first_float = [], 0, None
    generated_pixels = 0
    torch.cuda.synchronize() if device == 'cuda' else None
    sample_start = time.perf_counter()
    for start in range(0, config['eval_samples'], 128):
        noise = torch.randn(min(128, config['eval_samples']-start), *SHAPES[dataset], generator=rng).to(device)
        points = sample(model, noise, solver_steps)
        assert torch.isfinite(points).all()
        outside += int(((points < -1) | (points > 1)).sum())
        generated_pixels += points.numel()
        if first_float is None:
            first_float = points.cpu()
        images.append(to_uint8(points).cpu())
    images = torch.cat(images)
    torch.cuda.synchronize() if device == 'cuda' else None
    sampling_seconds = time.perf_counter() - sample_start
    noise = torch.randn(len(first_float), *SHAPES[dataset], generator=torch.Generator().manual_seed(config['eval_seed']))
    replay = sample(model, noise.to(device), solver_steps).cpu()
    assert torch.equal(replay, first_float)
    np.save(output / 'images.npy', images.numpy())
    np.save(output / 'first_float.npy', first_float.numpy())
    save_image(images[:64].float()/255, output / 'samples.png', nrow=8)
    result = dict(method='flow_matching', dataset=dataset, seed=seed, step=step,
                  signature=signature, samples=len(images), solver='explicit_midpoint',
                  solver_steps=solver_steps, nfe_per_image=2*solver_steps,
                  sampling_seconds=sampling_seconds, raw_pixels_outside_range_fraction=outside/generated_pixels,
                  generated_uint8_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                  first_batch_float_replay_exact=True, weights='raw')
    del model, replay
    if dataset == 'mnist':
        from .mnist_eval import Evaluator, predict, distribution_metrics
        evaluator = Evaluator(dataset_root, config, 'flow_matching', seed, device)
        prob = predict(evaluator.classifier, images, device).numpy()
        result.update(distribution_metrics(prob, evaluator.target))
        result['evaluation_signature'] = evaluator.signature
        np.savez_compressed(output / 'predictions.npz', probabilities=prob)
        grid = torch.zeros(100, 1, 28, 28)
        for digit in range(10):
            ids = np.flatnonzero(prob.argmax(1) == digit)[:10]
            if len(ids):
                grid[digit*10:digit*10+len(ids)] = images[ids].float()/255
        save_image(grid, output / 'by_predicted_digit.png', nrow=10)
    else:
        from .cifar_deficit import load_classifier, WEIGHTS
        from .cifar_class_representation import predict, summarize, gallery, REPO
        from .integrity_eval import extractor, extract, metrics
        cache = dataset_root / 'integrity-v1'
        result.update(classifiers={}, real_classifiers={}, classifier_weight_hashes=WEIGHTS,
                      classifier_repository=REPO,
                      reference_manifest=json.loads((cache/'manifest.json').read_text()))
        real = torch.from_numpy(np.load(cache / 'test_uint8.npy'))
        labels = np.load(cache / 'test_labels.npy')
        predictions = []
        for name in WEIGHTS:
            classifier = load_classifier(name, dataset_root.parent/'class-representation-weights', device)
            p = predict(classifier, images, device)
            rp = predict(classifier, real, device)
            reference = summarize(rp)
            reference['accuracy'] = float((rp.argmax(1) == labels).mean())
            assert reference['accuracy'] > .9
            result['real_classifiers'][name] = reference
            result['classifiers'][name] = summarize(p)
            predictions.append(p.argmax(1))
            np.savez_compressed(output/f'{name}.npz', probabilities=p)
            gallery(images, p, output/f'{name}.png', f'Flow matching, {step:,} updates, midpoint{solver_steps}')
            del classifier
        result['classifier_agreement'] = float((predictions[0] == predictions[1]).mean())
        ext = extractor(device)
        features = extract(images.numpy(), ext, device)
        del ext
        heldout = torch.load(cache/'test_features.pt', map_location='cpu', weights_only=True)
        train = torch.load(cache/'train_features.pt', map_location='cpu', weights_only=True)
        ids = np.load(cache/'reference_ids.npz')['train_metrics']
        result['heldout_metrics'] = metrics(features, heldout, device)
        result['train_metrics'] = metrics(features, train[ids], device)
        result['reference'] = 'official CIFAR10 test and original fixed train10k subset; neither selects training'
    result['evaluation_seconds'] = time.perf_counter()-began
    (output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(dest, result)
    if commit:
        commit()
    return result
