"""Exact continuations of image-flow pilots; evaluate every 10k updates."""
import json
import shutil
import re
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('paired-discriminator-image-flow-extension')
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




@app.function(image=image, gpu='A10', cpu=4, memory=16384, volumes=volumes,
              timeout=10800, retries=1, max_containers=6, scaledown_window=2)
def extend(run_id: str, dataset: str, seed: int, endpoint: int):
    import numpy as np
    import torch
    from paired_discriminator.image_flow import train
    from paired_discriminator.image_flow_eval import evaluate, preview
    from paired_discriminator.mnist import atomic_json
    validate(run_id)
    mnist_volume.reload(); cifar_volume.reload(); run_volume.reload()
    original = Path('/flow-runs/image-flow-v1')/f'{dataset}_seed{seed}'
    output = Path('/flow-runs')/run_id/f'{dataset}_seed{seed}'
    if not output.exists():
        shutil.copytree(original, output)
    config = json.loads(Path('/project/configs/image-flow.json').read_text())
    checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
    atomic_json(output/'continuation.json', dict(parent='image-flow-v1', start_step=10000,
                requested_endpoint=endpoint, resumed_step=checkpoint['step']))
    actual_gpu = torch.cuda.get_device_name()
    if checkpoint['signature']['gpu'] != actual_gpu:
        from paired_discriminator.image_flow import make_model, sample, SHAPES
        from paired_discriminator.mnist import configure
        configure('cuda')
        # Only permit the two Ampere A10 names Modal assigns to this GPU tier.
        assert {checkpoint['signature']['gpu'], actual_gpu} <= {'NVIDIA A10', 'NVIDIA A10G'}
        pilot = torch.load(original/'velocity_010000.pt', map_location='cpu', weights_only=False)
        model = make_model(dataset, config, seed).cuda()
        model.load_state_dict(pilot['model'])
        noise = torch.randn(128, *SHAPES[dataset], generator=torch.Generator().manual_seed(config['eval_seed']))
        current = sample(model, noise.cuda(), 64).cpu().numpy()
        reference = np.load(original/'eval_010000_midpoint64/first_float.npy')
        error = float(np.max(np.abs(current-reference)))
        assert error < .001, error
        atomic_json(output/'hardware_migration.json', dict(previous_gpu=checkpoint['signature']['gpu'],
            current_gpu=actual_gpu, step=checkpoint['step'], pilot_first_batch_max_abs_error=error,
            note='Optimizer/model/RNG restored unchanged; cross-GPU continuation is not claimed bitwise identical.'))
        checkpoint['signature']['gpu'] = actual_gpu
        torch.save(checkpoint, output/'latest.pt')
        del model
    data_root = Path('/mnist-data/mnist' if dataset == 'mnist' else '/data/cifar10')
    data = torch.from_numpy(np.load(data_root/'train_uint8.npy'))
    for step in range(20000, endpoint+1, 10000):
        if checkpoint['step'] > step:
            continue
        status = train(config, dataset, seed, step, output, data, commit=run_volume.commit,
                       on_checkpoint=lambda model, s, out: preview(model, dataset, config, s, out))
        status.update(phase='evaluating', target_steps=endpoint)
        atomic_json(output/'status.json', status); run_volume.commit()
        evaluate(output, data_root, step, 64, commit=run_volume.commit)
        print(dataset, seed, 'evaluated', step, flush=True)
    status.update(phase='complete', target_steps=endpoint)
    atomic_json(output/'status.json', status); run_volume.commit()
    return status

@app.function(image=image, volumes={'/flow-runs': run_volume}, timeout=14400)
def coordinate(run_id: str, endpoint: int):
    from paired_discriminator.mnist import atomic_json
    (Path('/flow-runs')/run_id).mkdir(parents=True, exist_ok=True)
    run_volume.commit()
    jobs = [('mnist', s) for s in range(5)] + [('cifar10', 0)]
    calls = [(d,s,extend.spawn(run_id,d,s,endpoint)) for d,s in jobs]
    atomic_json(Path('/flow-runs')/run_id/'launch.json', dict(run_id=run_id, endpoint=endpoint,
        parent='image-flow-v1', calls=[dict(dataset=d, seed=s, call_id=c.object_id) for d,s,c in calls]))
    run_volume.commit()
    return [c.get() for d,s,c in calls]

@app.local_entrypoint()
def main(run_id: str='image-flow-50k-v1', endpoint: int=50000):
    call = coordinate.spawn(run_id, endpoint)
    print('COORDINATOR', call.object_id, flush=True)
    destination=ROOT/'results'/run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'dispatch.json').write_text(json.dumps(dict(call_id=call.object_id,endpoint=endpoint),indent=2)+'\n')
    print(call.get(), flush=True)

@app.function(image=image, gpu='A10', cpu=4, memory=16384, volumes=volumes, timeout=1800, scaledown_window=2)
def solver_check(run_id: str, dataset: str, step: int):
    from paired_discriminator.image_flow_eval import evaluate
    run_volume.reload(); mnist_volume.reload(); cifar_volume.reload()
    return evaluate(Path('/flow-runs')/run_id/f'{dataset}_seed0',
                    Path('/mnist-data/mnist' if dataset=='mnist' else '/data/cifar10'),
                    step, 128, commit=run_volume.commit)

@app.local_entrypoint()
def check(run_id: str='image-flow-50k-v1', dataset: str='mnist', step: int=50000):
    print(solver_check.remote(run_id, dataset, step))

@app.local_entrypoint()
def cifar_more(run_id: str='image-flow-50k-v1', endpoint: int=100000):
    call=extend.spawn(run_id, 'cifar10', 0, endpoint)
    print('CIFAR CONTINUATION',call.object_id,flush=True)
    print(call.get(),flush=True)
