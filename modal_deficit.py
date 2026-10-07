"""Four-arm unconditional CIFAR deficit pilot, with held-out evaluation every 10k."""
from pathlib import Path
import json
import modal
import re

ROOT = Path(__file__).resolve().parent
data_volume = modal.Volume.from_name("paired-discriminator-cifar10-data", create_if_missing=True)
run_volume = modal.Volume.from_name("paired-discriminator-cifar10-runs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(uv_project_dir=str(ROOT), extras=["cifar"])
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCH_HOME": "/data/torch",
               "PYTHONPATH": "/root"})
         .add_local_dir(ROOT / "src" / "paired_discriminator", "/root/paired_discriminator")
         .add_local_dir(ROOT / "configs", "/project/configs")
         .add_local_file(ROOT / "pyproject.toml", "/project/pyproject.toml")
         .add_local_file(ROOT / "uv.lock", "/project/uv.lock"))


def validate_run_id(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Use a short alphanumeric run ID, with hyphens/underscores")




app = modal.App('paired-discriminator-cifar10-deficit')
METHODS = ('vanilla', 'paired', 'vanilla_deficit', 'paired_deficit')

@app.function(image=image, gpu='A10', cpu=4, memory=16384,
              volumes={'/data':data_volume,'/runs':run_volume}, timeout=1800, scaledown_window=2)
def verify(run_id: str):
    import numpy as np
    import torch
    from paired_discriminator.cifar_deficit import train, load_classifier
    from paired_discriminator.cifar import train as old_train, atomic_json
    from paired_discriminator.cifar_integrity import prepare_data
    from paired_discriminator.integrity_eval import prepare
    validate_run_id(run_id)
    prepare_data('/data/cifar10')
    prepare('/data/cifar10')
    data_volume.commit()
    config=json.loads(Path('/project/configs/cifar10-deficit.json').read_text())
    config.update(batch_size=4,log_every=1,checkpoint_every=2,eval_every=2)
    all_data=torch.from_numpy(np.load('/data/cifar10/train_uint8.npy'))
    all_labels=torch.from_numpy(np.load('/data/cifar10/train_labels.npy'))
    ids=torch.cat([(all_labels==k).nonzero().flatten()[:2] for k in range(10)])
    data, labels=all_data[ids],all_labels[ids]
    classifier=load_classifier('cifar10_resnet56','/data/class-representation-weights')
    root=Path('/runs')/run_id/'verification'
    for method in METHODS:
        train(config,method,0,4,root/method/'full',data,labels,classifier)
        train(config,method,0,2,root/method/'split',data,labels,classifier)
        train(config,method,0,4,root/method/'split',data,labels,classifier)
        a=torch.load(root/method/'full/latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(root/method/'split/latest.pt',map_location='cpu',weights_only=False)
        torch.testing.assert_close(a['ema'],b['ema'],rtol=0,atol=0)
        for key in ('generator','discriminator','rngs'):
            for name in a[key]:
                torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
        if not method.endswith('_deficit'):
            old_train(config,method,0,4,root/method/'original',data)
            old=torch.load(root/method/'original/latest.pt',map_location='cpu',weights_only=False)
            for key in ('generator','discriminator','rngs'):
                for name in a[key]:
                    torch.testing.assert_close(a[key][name],old[key][name],rtol=0,atol=0)
        # Deficit sampling cannot consume G or noise-D RNG streams.
        reference=torch.load(root/'vanilla/full/latest.pt',map_location='cpu',weights_only=False)
        for key in ('noise_d','noise_g','real_g','slots'):
            torch.testing.assert_close(a['rngs'][key],reference['rngs'][key],rtol=0,atol=0)
    result={'cuda_exact_resume':True,'uniform_matches_original':True,'G_and_noise_RNG_preserved':True,'methods':METHODS,'steps':4,'split_at':2}
    atomic_json(root/'verification.json',result)
    run_volume.commit()
    return result

@app.function(image=image,gpu='A10',cpu=4,memory=16384,
              volumes={'/data':data_volume,'/runs':run_volume},timeout=21600,retries=1,
              max_containers=4,scaledown_window=2)
def run_arm(run_id: str,method: str,seed: int,steps: int):
    import shutil
    import numpy as np
    import torch
    import paired_discriminator.cifar as original
    from paired_discriminator.cifar_deficit import train,load_classifier
    from paired_discriminator.cifar_deficit_eval import evaluate
    validate_run_id(run_id)
    if method not in METHODS or steps%10000 or steps<=0:
        raise ValueError('Unknown arm or invalid endpoint')
    config=json.loads(Path('/project/configs/cifar10-deficit.json').read_text())
    output=Path('/runs')/run_id/f'{method}_seed{seed}'
    output.mkdir(parents=True,exist_ok=True)
    for name in ('pyproject.toml','uv.lock'):
        source=Path('/project')/name
        destination=output/name
        if destination.exists() and destination.read_bytes()!=source.read_bytes():
            raise ValueError('Dependency manifest changed')
        shutil.copy2(source,destination)
    shutil.copy2(original.__file__,output/'original_cifar_source.py')
    data=torch.from_numpy(np.load('/data/cifar10/train_uint8.npy'))
    labels=torch.from_numpy(np.load('/data/cifar10/train_labels.npy'))
    classifier=load_classifier('cifar10_resnet56','/data/class-representation-weights')
    # Complete a checkpoint's evaluation before advancing this arm. Resume skips finished segments.
    for step in range(10000,steps+1,10000):
        latest=output/'status.json'
        current=json.loads(latest.read_text())['step'] if latest.exists() else 0
        if current<=step:
            train(config,method,seed,step,output,data,labels,classifier,commit=run_volume.commit)
        result=evaluate(output,'/data/cifar10',step,commit=run_volume.commit)
        print(method,step,json.dumps(result['heldout_metrics']),flush=True)
    return {'method':method,'seed':seed,'step':steps,'output':str(output)}

@app.local_entrypoint()
def main(run_id: str='cifar10-deficit-v1',steps: int=100000,seed: int=0):
    validate_run_id(run_id)
    print('Verification:',verify.remote(run_id),flush=True)
    calls=[(method,run_arm.spawn(run_id,method,seed,steps)) for method in METHODS]
    for method,call in calls:
        print(method,call.get(),flush=True)
