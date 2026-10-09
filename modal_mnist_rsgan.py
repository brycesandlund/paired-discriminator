"""Experiment 39: matched RSGAN addition to the frozen MNIST protocol."""
from pathlib import Path
import json
import modal
import re
ROOT=Path(__file__).resolve().parent
app=modal.App('paired-discriminator-mnist-rsgan')
data_volume=modal.Volume.from_name('paired-discriminator-mnist-data',create_if_missing=True)
run_volume=modal.Volume.from_name('paired-discriminator-mnist-runs',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
       .uv_sync(uv_project_dir=str(ROOT),extras=['cifar'])
       .env({'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':'/root'})
       .add_local_dir(ROOT/'src/paired_discriminator','/root/paired_discriminator')
       .add_local_dir(ROOT/'configs','/project/configs')
       .add_local_file(ROOT/'pyproject.toml','/project/pyproject.toml')
       .add_local_file(ROOT/'uv.lock','/project/uv.lock'))


def validate(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',run_id):raise ValueError('Invalid run id')

RUN='mnist-rsgan-v1'

@app.function(image=image,gpu='A10',cpu=4,memory=8192,volumes={'/data':data_volume,'/runs':run_volume},
              timeout=7200,retries=1,max_containers=4,scaledown_window=2)
def train_job(run_id:str,method:str,seed:int,steps:int):
    import shutil
    import numpy as np
    import torch
    from paired_discriminator.mnist import train
    from paired_discriminator.mnist_eval import Evaluator
    validate(run_id);data_volume.reload();run_volume.reload()
    config=json.loads(Path('/project/configs/mnist.json').read_text())
    out=Path('/runs')/run_id/f'{method}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    for name in ('pyproject.toml','uv.lock'):
        dest=out/name
        if dest.exists() and dest.read_bytes()!=(Path('/project')/name).read_bytes():raise ValueError('Dependency manifest changed')
        shutil.copy2(Path('/project')/name,dest)
    evaluator=Evaluator('/data/mnist',config,method,seed)
    shutil.copy2('/data/mnist/classifier-v1/report.json',out/'classifier_report.json')
    data=torch.from_numpy(np.load('/data/mnist/train_uint8.npy'))
    return train(config,method,seed,steps,out,data,evaluator=evaluator,evaluation_signature=evaluator.signature,commit=run_volume.commit)


@app.function(image=image,gpu='A10',cpu=4,memory=8192,volumes={'/data':data_volume,'/runs':run_volume},timeout=900,scaledown_window=2)
def verify_rsgan(expected_classifier: str):
    import numpy as np
    import torch
    from paired_discriminator import mnist
    from paired_discriminator.mnist_eval import Evaluator
    data_volume.reload();run_volume.reload()
    config=json.loads(Path('/project/configs/mnist.json').read_text())
    config.update(log_every=10,checkpoint_every=20,eval_every=20,eval_samples=128)
    data=torch.from_numpy(np.load('/data/mnist/train_uint8.npy')[:320])
    root=Path('/runs')/RUN/'verification'
    evaluator=Evaluator('/data/mnist',config,'rsgan',0)
    assert evaluator.signature['classifier_sha256']==expected_classifier
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    g,d=mnist.models(config,'rsgan',0);gv,dv=mnist.models(config,'vanilla',0)
    assert equal(g.state_dict(),gv.state_dict()) and equal(d.state_dict(),dv.state_dict())
    kw=dict(evaluator=evaluator,evaluation_signature=evaluator.signature)
    mnist.train(config,'rsgan',0,40,root/'full',data,**kw)
    if not (root/'split/latest.pt').exists():
        mnist.train(config,'rsgan',0,20,root/'split',data,**kw)
    mnist.train(config,'rsgan',0,40,root/'split',data,**kw)
    mnist.train(config,'rsgan',0,40,root/'no_eval',data)
    a=torch.load(root/'full/latest.pt',weights_only=False)
    for folder in ('split','no_eval'):
        b=torch.load(root/folder/'latest.pt',weights_only=False)
        for key in ('generator','discriminator','optimizer_g','optimizer_d','rngs'):assert equal(a[key],b[key]),key
        assert [(x['d_loss'],x['g_loss']) for x in a['rows']]==[(x['d_loss'],x['g_loss']) for x in b['rows']]
    for step in (20,40):
        a=json.loads((root/'full'/f'metrics_{step:06d}.json').read_text())
        b=json.loads((root/'split'/f'metrics_{step:06d}.json').read_text())
        assert a['generated_uint8_sha256']==b['generated_uint8_sha256'] and a['digit_counts']==b['digit_counts']
    result=dict(cuda_exact_resume=True,optimizer_and_rng_exact=True,evaluation_neutral=True,initial_g_and_d_match_vanilla=True,classifier_sha256=expected_classifier)
    mnist.atomic_json(root/'verification.json',result);run_volume.commit();return result

@app.function(image=image,cpu=1,memory=1024,volumes={'/runs':run_volume},timeout=21600)
def coordinate(expected_classifier: str):
    from paired_discriminator.mnist import atomic_json
    run_volume.reload();root=Path('/runs')/RUN;root.mkdir(parents=True,exist_ok=True)
    path=root/'dispatch.json'
    if path.exists():
        dispatch=json.loads(path.read_text())
        assert len(dispatch)==5 and [x['seed'] for x in dispatch]==list(range(5)), 'Partial dispatch: inspect existing calls before repair'
    else:
        verification=verify_rsgan.remote(expected_classifier)
        atomic_json(root/'verification.json',verification);run_volume.commit()
        dispatch=[]
        for seed in range(5):
            call=train_job.spawn(RUN,'rsgan',seed,50000)
            dispatch.append(dict(seed=seed,method='rsgan',steps=50000,call_id=call.object_id))
            atomic_json(path,dispatch);run_volume.commit()
    results=[];errors=[]
    for item in dispatch:
        try:results.append(dict(seed=item['seed'],status=modal.FunctionCall.from_id(item['call_id']).get()))
        except Exception as error:errors.append(dict(seed=item['seed'],error=str(error)))
        atomic_json(root/'results.json',dict(results=results,errors=errors));run_volume.commit()
    atomic_json(root/'phase_status.json',dict(phase='complete',errors=errors));run_volume.commit()
    return dict(completed=len(results),errors=errors)

@app.local_entrypoint()
def launch_rsgan():
    expected=json.loads((ROOT/'results/mnist-v1/classifier/report.json').read_text())['checkpoint_sha256']
    call=coordinate.spawn(expected)
    out=ROOT/'results'/RUN;out.mkdir(parents=True,exist_ok=True)
    (out/'launch.json').write_text(json.dumps(dict(call_id=call.object_id,steps=50000,seeds=list(range(5)),classifier_sha256=expected),indent=2)+'\n')
    print('COORDINATOR',call.object_id,flush=True);print(call.get(),flush=True)

@app.function(image=image,cpu=1,memory=2048,volumes={'/runs':run_volume},timeout=600)
def bundle():
    import io,tarfile
    run_volume.reload();root=Path('/runs')/RUN;buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.png','.py','.csv','.toml','.lock'):
                tar.add(p,arcname=str(p.relative_to(root)))
    return buffer.getvalue()

@app.local_entrypoint()
def collect_rsgan():
    import io,tarfile
    payload=bundle.remote();out=ROOT/'results'/RUN;out.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:tar.extractall(out,filter='data')
    print('Collected',len(payload),'bytes',flush=True)
