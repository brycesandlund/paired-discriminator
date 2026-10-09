"""Bounded paired-discriminator architecture ablations; no generator changes."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
RUN = 'cifar-pair-arch-v1'
app = modal.App('paired-discriminator-cifar-pair-architecture')
data_volume = modal.Volume.from_name('paired-discriminator-cifar10-data')
run_volume = modal.Volume.from_name('paired-discriminator-cifar-pair-architecture', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12').uv_sync(uv_project_dir=str(ROOT), extras=['cifar'])
         .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TORCH_HOME': '/data/torch', 'PYTHONPATH': '/root'})
         .add_local_dir(ROOT / 'src/paired_discriminator', '/root/paired_discriminator'))

@app.function(image=image, gpu='A10', cpu=4, memory=16384, timeout=10800, retries=1,
              volumes={'/data': data_volume, '/runs': run_volume}, max_containers=4, scaledown_window=2)
def job(spec: dict):
    import numpy as np
    import torch
    from paired_discriminator import cifar_pair_arch as arch, cifar, cifar_tune
    run_volume.reload()
    root = Path('/runs') / RUN
    name, seed = spec['architecture'], spec.get('seed', 0)
    if name == 'symmetric_concat':
        from paired_discriminator import cifar_pair_symmetry as arch
    if spec.get('kind') == 'verify':
        data = torch.from_numpy(np.load('/data/cifar10/train_uint8.npy')[:16].copy())
        base = root / 'verify' / name
        a = arch.train(name, seed, 4, base/'full', data, batch=4)
        arch.train(name, seed, 2, base/'split', data, batch=4)
        b = arch.train(name, seed, 4, base/'split', data, batch=4)
        for key in ('model','discriminator','ema','rngs','optimizer_g','optimizer_d'):
            assert arch.state_equal(a[key], b[key]), key
        if name == 'concat':
            old = cifar_tune.train('paired', seed, cifar_tune.configs('paired')[0] | {'batch_size':4},
                                   4, base/'old', data)
            for key in ('model','discriminator','ema','optimizer_g','optimizer_d'):
                assert arch.state_equal(a[key], old[key]), key
        result = dict(architecture=name, exact_cuda_resume=True,
                      original_control_equivalent=name=='concat')
        cifar.atomic_json(base/'verification.json', result); run_volume.commit()
        return result
    out = root / f'{name}_seed{seed}'
    data = torch.from_numpy(np.load('/data/cifar10/train_uint8.npy'))
    records = []
    for step in spec.get('eval_steps', [10000, 20000, 50000]):
        existing = json.loads((out/'status.json').read_text())['step'] if (out/'status.json').exists() else 0
        if existing <= step:
            arch.train(name, seed, step, out, data, commit=run_volume.commit)
        ck = out / f'checkpoint_{step:06d}.pt'
        for weights in ('model', 'ema'):
            result = cifar_tune.evaluate(ck, 'paired', seed, weights,
                out/f'eval_{step:06d}_{weights}', noise_seed=spec.get('noise_seed', 99374), commit=run_volume.commit)
            records.append(dict(architecture=name, step=step, seed=seed, weights=weights, metrics=result))
        diagnostic_path = out / f'diagnostics_{step:06d}.json'
        if not diagnostic_path.exists():
            cifar.atomic_json(diagnostic_path, arch.diagnostics(ck, name, seed, data))
        status = json.loads((out/'status.json').read_text())
        status.update(phase='complete' if step==spec['steps'] else 'milestone', target=spec['steps'], evaluated_step=step)
        cifar.atomic_json(out/'status.json', status); run_volume.commit()
    return dict(spec=spec, records=records)

@app.function(image=image, volumes={'/runs':run_volume}, timeout=21600)
def coordinate(specs: list, phase: str, verify: bool | list[str]=False):
    from paired_discriminator import cifar_pair_arch as arch, cifar
    root = Path('/runs')/RUN; root.mkdir(parents=True, exist_ok=True)
    if (root/f'{phase}_dispatch.json').exists():
        raise RuntimeError('This phase was already dispatched; inspect its calls rather than duplicating it')
    if verify:
        names = verify if isinstance(verify,list) else arch.ARCHITECTURES
        calls = [job.spawn(dict(kind='verify',architecture=name)) for name in names]
        results = [call.get() for call in calls]
        cifar.atomic_json(root/('verification.json' if phase=='screen' else f'{phase}_verification.json'), results); run_volume.commit()
    calls = [job.spawn(spec) for spec in specs]
    cifar.atomic_json(root/f'{phase}_dispatch.json', dict(calls=[dict(spec=s, call_id=c.object_id) for s,c in zip(specs,calls)]))
    run_volume.commit(); results=[]; errors=[]
    for spec, call in zip(specs,calls):
        try: results.append(call.get())
        except Exception as error: errors.append(dict(spec=spec, error=str(error)))
        cifar.atomic_json(root/f'{phase}_results.json', dict(results=results, errors=errors)); run_volume.commit()
    cifar.atomic_json(root/f'{phase}_status.json',dict(phase='complete',errors=errors)); run_volume.commit()
    return dict(jobs=len(results), errors=errors)

@app.local_entrypoint()
def main():
    from paired_discriminator.cifar_pair_arch import ARCHITECTURES
    specs=[dict(architecture=name,seed=0,steps=50000,eval_steps=[10000,20000,50000],noise_seed=99374)
           for name in ARCHITECTURES]
    call=coordinate.spawn(specs,'screen',True)
    out=ROOT/'results'/RUN;out.mkdir(parents=True,exist_ok=True)
    (out/'launch.json').write_text(json.dumps(dict(call_id=call.object_id,specs=specs),indent=2)+'\n')
    print('COORDINATOR',call.object_id,flush=True);print(call.get(),flush=True)

@app.local_entrypoint()
def phase(specs_path: str, name: str):
    specs=json.loads(Path(specs_path).read_text())
    checks=['symmetric_concat'] if any(s['architecture']=='symmetric_concat' for s in specs) else False
    call=coordinate.spawn(specs,name,checks)
    out=ROOT/'results'/RUN;out.mkdir(parents=True,exist_ok=True)
    (out/f'{name}_launch.json').write_text(json.dumps(dict(call_id=call.object_id),indent=2)+'\n')
    print('COORDINATOR',call.object_id,flush=True);print(call.get(),flush=True)

@app.local_entrypoint()
def single(specs_path: str):
    call=job.spawn(json.loads(Path(specs_path).read_text()))
    print('JOB',call.object_id,flush=True);print(call.get(),flush=True)


@app.function(image=image, cpu=1, memory=1024, volumes={'/runs':run_volume}, timeout=21600)
def resume_collection(specs: list, phase: str):
    """Rejoin saved calls after coordinator failure; never dispatch GPU work."""
    from paired_discriminator import cifar
    run_volume.reload()
    root=Path('/runs')/RUN
    dispatch=json.loads((root/f'{phase}_dispatch.json').read_text())
    assert [c['spec'] for c in dispatch['calls']]==specs, 'Saved dispatch differs from frozen specs'
    cifar.atomic_json(root/f'{phase}_recovery.json',dict(
        reason='Original coordinator was preempted, then its duplicate-dispatch guard stopped the restarted coordinator.',
        action='Rejoin existing saved FunctionCall IDs only; no new training jobs.',
        calls=[c['call_id'] for c in dispatch['calls']]))
    run_volume.commit()
    results=[];errors=[]
    for item in dispatch['calls']:
        try:results.append(modal.FunctionCall.from_id(item['call_id']).get())
        except Exception as error:errors.append(dict(spec=item['spec'],error=str(error)))
        cifar.atomic_json(root/f'{phase}_results.json',dict(results=results,errors=errors));run_volume.commit()
    cifar.atomic_json(root/f'{phase}_status.json',dict(phase='complete',errors=errors,
        coordinator_recovered=True,training_redispatched=False));run_volume.commit()
    return dict(jobs=len(results),errors=errors)


@app.local_entrypoint()
def recover(specs_path: str, name: str):
    specs=json.loads(Path(specs_path).read_text())
    call=resume_collection.spawn(specs,name)
    out=ROOT/'results'/RUN
    (out/f'{name}_recovery_launch.json').write_text(json.dumps(dict(call_id=call.object_id),indent=2)+'\n')
    print('RECOVERY_COORDINATOR',call.object_id,flush=True);print(call.get(),flush=True)


@app.function(image=image, cpu=1, memory=1024, volumes={'/runs':run_volume}, timeout=300)
def archive_artifacts():
    """Bundle small artifacts to avoid hundreds of separate volume downloads."""
    import io
    import tarfile
    run_volume.reload()
    root=Path('/runs')/RUN
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.png','.py'):
                archive.add(path,arcname=str(path.relative_to(root)))
    return buffer.getvalue()


@app.local_entrypoint()
def collect():
    import io
    import tarfile
    payload=archive_artifacts.remote()
    out=ROOT/'results'/RUN;out.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        archive.extractall(out,filter='data')
    print('Collected artifact bundle:',len(payload),'bytes',flush=True)
