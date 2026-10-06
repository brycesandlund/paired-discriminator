"""Ordinary MNIST, unconditional vanilla versus paired, five seeds."""
from pathlib import Path
import json
import re
import modal

ROOT=Path(__file__).resolve().parent
app=modal.App('paired-discriminator-mnist')
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


@app.function(image=image,gpu='A10',cpu=4,memory=8192,volumes={'/data':data_volume},timeout=1800,scaledown_window=2)
def prepare():
    from paired_discriminator.mnist import prepare_data
    from paired_discriminator.mnist_eval import prepare_classifier
    print(json.dumps(prepare_data('/data/mnist')),flush=True)
    result=prepare_classifier('/data/mnist')
    assert max(r['validation_accuracy'] for r in result['history'])>=.98
    data_volume.commit()
    return result


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
def verify():
    import numpy as np
    import torch
    from paired_discriminator.mnist import train,models,atomic_json
    from paired_discriminator.mnist_eval import Evaluator
    data_volume.reload();run_volume.reload()
    config=json.loads(Path('/project/configs/mnist.json').read_text())
    config.update(log_every=10,checkpoint_every=20,eval_every=20,eval_samples=128)
    data=torch.from_numpy(np.load('/data/mnist/train_uint8.npy')[:320])
    root=Path('/runs/mnist-cuda-resume-check-v1')
    gs=[models(config,m,0)[0] for m in ('vanilla','paired')]
    for k,v in gs[0].state_dict().items():torch.testing.assert_close(v,gs[1].state_dict()[k],rtol=0,atol=0)
    for method in ('vanilla','paired'):
        evaluator=Evaluator('/data/mnist',config,method,0)
        kw={'evaluator':evaluator,'evaluation_signature':evaluator.signature}
        train(config,method,0,40,root/method/'full',data,**kw)
        train(config,method,0,20,root/method/'split',data,**kw)
        train(config,method,0,40,root/method/'split',data,**kw)
        train(config,method,0,40,root/method/'no_eval',data)
        a=torch.load(root/method/'full/latest.pt',weights_only=False)
        for directory in ('split','no_eval'):
            b=torch.load(root/method/directory/'latest.pt',weights_only=False)
            for key in ('generator','discriminator','rngs'):
                for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
            assert [(r['d_loss'],r['g_loss']) for r in a['rows']]==[(r['d_loss'],r['g_loss']) for r in b['rows']]
        for step in (20,40):
            a=json.loads((root/method/'full'/f'metrics_{step:06d}.json').read_text())
            b=json.loads((root/method/'split'/f'metrics_{step:06d}.json').read_text())
            assert a['generated_uint8_sha256']==b['generated_uint8_sha256']
            assert a['digit_counts']==b['digit_counts']
    result={'cuda_exact_resume':True,'evaluation_does_not_change_training':True,'same_generator_initialization':True,'methods':['vanilla','paired'],'steps':40,'split_at':20}
    atomic_json(root/'verification.json',result);run_volume.commit();return result


@app.local_entrypoint()
def main(run_id:str='mnist-v1',steps:int=50000):
    validate(run_id)
    classifier=prepare.remote()
    print('classifier',json.dumps({k:classifier[k] for k in ('test_accuracy','selected_epoch','checkpoint_sha256')}),flush=True)
    print('verification',json.dumps(verify.remote()),flush=True)
    calls=[(method,seed,train_job.spawn(run_id,method,seed,steps)) for seed in range(5) for method in ('vanilla','paired')]
    for method,seed,call in calls:print('finished',method,seed,json.dumps(call.get()),flush=True)
