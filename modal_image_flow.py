"""Launch native-image flow pilots, with existing MNIST and CIFAR evaluators."""

import io
import json
import re
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('paired-discriminator-image-flow')
mnist_volume = modal.Volume.from_name('paired-discriminator-mnist-data')
cifar_volume = modal.Volume.from_name('paired-discriminator-cifar10-data')
run_volume = modal.Volume.from_name('paired-discriminator-image-flow-runs', create_if_missing=True)
volumes = {'/mnist-data': mnist_volume, '/data': cifar_volume, '/flow-runs': run_volume}
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(uv_project_dir=str(ROOT), extras=['cifar'])
         .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TORCH_HOME': '/data/torch', 'PYTHONPATH': '/root'})
         .add_local_dir(ROOT/'src/paired_discriminator', '/root/paired_discriminator')
         .add_local_dir(ROOT/'configs', '/project/configs')
         .add_local_file(ROOT/'pyproject.toml', '/project/pyproject.toml')
         .add_local_file(ROOT/'uv.lock', '/project/uv.lock'))


def validate(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')


@app.function(image=image, gpu='A10', cpu=4, memory=8192, volumes=volumes, timeout=1800, scaledown_window=2)
def prepare_and_verify(run_id: str):
    import numpy as np
    import torch
    from paired_discriminator.image_flow import train, sample
    from paired_discriminator.mnist import atomic_json, prepare_data
    from paired_discriminator.mnist_eval import prepare_classifier
    from paired_discriminator.integrity_eval import prepare as prepare_cifar_metrics
    from paired_discriminator.cifar_deficit import WEIGHTS, load_classifier
    validate(run_id)
    mnist_volume.reload(); cifar_volume.reload(); run_volume.reload()
    prepare_data('/mnist-data/mnist')
    classifier = prepare_classifier('/mnist-data/mnist')
    assert classifier['test_accuracy'] > .98
    assert Path('/data/cifar10/train_uint8.npy').exists()
    prepare_cifar_metrics('/data/cifar10')
    for name in WEIGHTS:
        model = load_classifier(name, '/data/class-representation-weights')
        del model
    mnist_volume.commit(); cifar_volume.commit()
    config = json.loads(Path('/project/configs/image-flow.json').read_text())
    config.update(log_every=10, checkpoint_every=10)
    root = Path('/flow-runs')/run_id/'verification'
    timings = {}
    for dataset, data_root in [('mnist', '/mnist-data/mnist'), ('cifar10', '/data/cifar10')]:
        data = torch.from_numpy(np.load(Path(data_root)/'train_uint8.npy')[:256].copy())
        def observer(model, step, output):
            noise = torch.zeros(2, *data.shape[1:], device='cuda')
            sample(model, noise, 2)
        status = train(config, dataset, 0, 20, root/dataset/'full', data)
        train(config, dataset, 0, 10, root/dataset/'split', data, on_checkpoint=observer)
        train(config, dataset, 0, 20, root/dataset/'split', data, on_checkpoint=observer)
        a = torch.load(root/dataset/'full/latest.pt', map_location='cpu', weights_only=False)
        b = torch.load(root/dataset/'split/latest.pt', map_location='cpu', weights_only=False)
        for key in ('model', 'ema', 'rngs'):
            assert all(torch.equal(a[key][k], b[key][k]) for k in a[key])
        for pid, state in a['optimizer']['state'].items():
            for key, value in state.items():
                assert torch.equal(value, b['optimizer']['state'][pid][key])
        assert [r['loss'] for r in a['rows']] == [r['loss'] for r in b['rows']]
        timings[dataset] = status
    result = dict(cuda_exact_resume=True, sampling_does_not_change_training=True,
                  classifier_sha256=classifier['checkpoint_sha256'], timings=timings)
    atomic_json(root/'verification.json', result)
    run_volume.commit()
    return result


@app.function(image=image, gpu='A10', cpu=4, memory=16384, volumes=volumes,
              timeout=7200, retries=1, max_containers=6, scaledown_window=2)
def train_job(run_id: str, dataset: str, seed: int, steps: int):
    import shutil
    import numpy as np
    import torch
    from paired_discriminator.image_flow import train
    from paired_discriminator.image_flow_eval import preview, evaluate
    from paired_discriminator.mnist import atomic_json
    validate(run_id)
    assert dataset in ('mnist', 'cifar10')
    mnist_volume.reload(); cifar_volume.reload(); run_volume.reload()
    config = json.loads(Path('/project/configs/image-flow.json').read_text())
    output = Path('/flow-runs')/run_id/f'{dataset}_seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock'):
        dest = output/name
        source = Path('/project')/name
        if dest.exists():
            assert dest.read_bytes() == source.read_bytes()
        shutil.copy2(source, dest)
    data_root = Path('/mnist-data/mnist' if dataset == 'mnist' else '/data/cifar10')
    data = torch.from_numpy(np.load(data_root/'train_uint8.npy'))
    status = train(config, dataset, seed, steps, output, data, commit=run_volume.commit,
                   on_checkpoint=lambda model, step, out: preview(model, dataset, config, step, out))
    status['phase'] = 'evaluating'
    atomic_json(output/'status.json', status); run_volume.commit()
    solver_steps = [config['solver_steps']] + ([config['solver_check_steps']] if seed == 0 else [])
    evaluations = []
    for solver in solver_steps:
        result = evaluate(output, data_root, steps, solver, commit=run_volume.commit)
        evaluations.append(dict(solver_steps=solver, evaluation_seconds=result['evaluation_seconds'],
                                sampling_seconds=result['sampling_seconds']))
    status.update(phase='complete', evaluations=evaluations)
    atomic_json(output/'status.json', status); run_volume.commit()
    return status


@app.function(image=image, volumes={'/flow-runs': run_volume}, timeout=300)
def bundle(run_id: str):
    validate(run_id); run_volume.reload()
    root = Path('/flow-runs')/run_id
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in root.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.png', '.py', '.toml', '.lock'):
                # Modal mounts can preserve epoch-zero source-file mtimes.
                archive.writestr(file.relative_to(root).as_posix(), file.read_bytes())
    return buffer.getvalue()


def save_bundle(run_id, data):
    destination = ROOT/'results'/run_id
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            assert (destination/member.filename).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)


@app.local_entrypoint()
def main(run_id: str = 'image-flow-v1', steps: int = 10000):
    validate(run_id)
    verification = prepare_and_verify.remote(run_id)
    print('verification', json.dumps(verification), flush=True)
    jobs = [('mnist', seed) for seed in range(5)] + [('cifar10', 0)]
    calls = [(dataset, seed, train_job.spawn(run_id, dataset, seed, steps)) for dataset, seed in jobs]
    destination = ROOT/'results'/run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'launch.json').write_text(json.dumps(dict(run_id=run_id, steps=steps,
        calls=[dict(dataset=d, seed=s, call_id=c.object_id) for d, s, c in calls], verification=verification), indent=2)+'\n')
    for dataset, seed, call in calls:
        print('completed', dataset, seed, json.dumps(call.get()), flush=True)
    save_bundle(run_id, bundle.remote(run_id))


@app.local_entrypoint()
def collect(run_id: str = 'image-flow-v1'):
    validate(run_id)
    save_bundle(run_id, bundle.remote(run_id))
