"""Experiment40: two fresh recipes, 20 A10 training-minutes each."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parent
RUN='cifar-budget20-v1'
app=modal.App('paired-discriminator-cifar-budget20')
data_volume=modal.Volume.from_name('paired-discriminator-cifar10-data')
run_volume=modal.Volume.from_name('paired-discriminator-cifar-budget',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12').uv_sync(uv_project_dir=str(ROOT),extras=['cifar'])
       .env({'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_HOME':'/data/torch','PYTHONPATH':'/root'})
       .add_local_dir(ROOT/'src/paired_discriminator','/root/paired_discriminator'))

@app.function(image=image,gpu='A10',cpu=4,memory=16384,volumes={'/data':data_volume,'/runs':run_volume},
              timeout=3600,max_containers=2,scaledown_window=2,retries=0)
def job(spec:dict):
    import time,copy,statistics
    import torch,numpy as np
    from paired_discriminator import cifar_budget as budget,cifar_tune as old,cifar,image_flow
    run_volume.reload();data_volume.reload();root=Path('/runs')/RUN
    data=torch.from_numpy(np.load('/data/cifar10/train_uint8.npy'))
    began=time.perf_counter()
    if spec['kind']=='verify':
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        checks=[]
        for method in ('paired','flow'):
            cfg=budget.config(method)|{'batch_size':4}
            out=root/'verification'/method
            a=budget.train(method,3,cfg,4,out/'full',data[:16])
            if not (out/'split/latest.pt').exists():budget.train(method,3,cfg,2,out/'split',data[:16])
            b=budget.train(method,3,cfg,4,out/'split',data[:16])
            c=old.train(method,3,cfg,4,out/'original',data[:16])
            for k in ('model','ema','discriminator','optimizer_g','optimizer_d','rngs'):
                assert equal(a[k],b[k]) and equal(a[k],c[k]),(method,k)
            checks.append(dict(method=method,exact_cuda_resume=True,original_recipe_parity=True))
        # Exercise the GAN's historical optimizer/EMA/RNG reset on CUDA.
        cfg=budget.config('paired')|{'batch_size':4,'reset_step':2};out=root/'verification/reset'
        a=budget.train('paired',3,cfg,4,out/'full',data[:16])
        old.train('paired',3,cfg,2,out/'base',data[:16])
        b=old.train('paired',3,cfg,2,out/'branch',data[:16],parent=out/'base/latest.pt')
        for k in ('model','ema','discriminator','optimizer_g','optimizer_d','rngs'):assert equal(a[k],b[k]),k
        result=dict(checks=checks,paired_reset_parity=True,gpu_seconds=time.perf_counter()-began)
        cifar.atomic_json(root/'verification.json',result);run_volume.commit();return result
    method=spec['method'];out=root/f'{method}_seed{spec["seed"]}'
    if (out/'completed.json').exists():return json.loads((out/'completed.json').read_text())
    cfg=budget.config(method)
    ck=budget.train(method,spec['seed'],cfg,1000000,out,data,budget_seconds=spec['budget_seconds'],commit=run_volume.commit)
    assert spec['budget_seconds']<=ck['training_seconds']<=spec['budget_seconds']+5,ck['training_seconds']
    checkpoint=out/f'checkpoint_{ck["step"]:06d}.pt'
    evaluation=old.evaluate(checkpoint,method,spec['seed'],'ema',out/'eval_ema',noise_seed=spec['noise_seed'],commit=run_volume.commit)
    # Same fixed batch and synchronized steady-state timing, separate from metrics.
    g,_=old.models(method,spec['seed']);g.load_state_dict(ck['ema']);g.cuda().eval()
    rng=torch.Generator().manual_seed(99575)
    z=torch.randn((128,3,32,32) if method=='flow' else (128,128),generator=rng).cuda()
    timings=[]
    with torch.no_grad():
        for i in range(7):
            torch.cuda.synchronize();t=time.perf_counter()
            x=image_flow.sample(g,z,64) if method=='flow' else g(z)
            torch.cuda.synchronize()
            if i>=2:timings.append(time.perf_counter()-t)
    result=dict(spec=spec,step=ck['step'],training_seconds=ck['training_seconds'],evaluation_seconds=evaluation['evaluation_seconds'],
        worker_seconds=time.perf_counter()-began,gpu=torch.cuda.get_device_name(),
        inference=dict(batch_size=128,batch_seconds=timings,median_seconds_per_image=statistics.median(timings)/128,
            network_evaluations=128 if method=='flow' else 1),evaluation=evaluation)
    cifar.atomic_json(out/'completed.json',result);run_volume.commit();return result

@app.function(image=image,cpu=1,memory=1024,volumes={'/runs':run_volume},timeout=7200)
def coordinate(protocol:dict):
    from paired_discriminator.cifar import atomic_json
    run_volume.reload();root=Path('/runs')/RUN;root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:atomic_json(root/'protocol.json',protocol);run_volume.commit()
    path=root/'dispatch.json'
    if path.exists():
        dispatch=json.loads(path.read_text());assert len(dispatch)==2,'Inspect partial dispatch; do not duplicate calls'
        assert [r['spec'] for r in dispatch]==protocol['specs']
    else:
        if not (root/'verification.json').exists():job.remote(dict(kind='verify'))
        run_volume.reload();assert (root/'verification.json').exists()
        dispatch=[]
        for spec in protocol['specs']:
            c=job.spawn(spec);dispatch.append(dict(spec=spec,call_id=c.object_id))
            atomic_json(path,dispatch);run_volume.commit()
    results=[];errors=[]
    for r in dispatch:
        try:results.append(modal.FunctionCall.from_id(r['call_id']).get())
        except Exception as e:errors.append(dict(spec=r['spec'],error=str(e)))
        atomic_json(root/'results.json',dict(results=results,errors=errors));run_volume.commit()
    atomic_json(root/'phase_status.json',dict(phase='complete',errors=errors));run_volume.commit()
    return dict(completed=len(results),errors=errors)

@app.local_entrypoint()
def launch():
    out=ROOT/'results'/RUN;protocol=json.loads((out/'protocol.json').read_text())
    c=coordinate.spawn(protocol)
    (out/'launch.json').write_text(json.dumps(dict(call_id=c.object_id),indent=2)+'\n')
    print('COORDINATOR',c.object_id,flush=True);print(c.get(),flush=True)

@app.function(image=image,cpu=1,memory=2048,volumes={'/runs':run_volume},timeout=600)
def bundle():
    import io,tarfile
    run_volume.reload();root=Path('/runs')/RUN;b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.py','.png'):t.add(p,arcname=str(p.relative_to(root)))
    return b.getvalue()

@app.local_entrypoint()
def collect():
    import io,tarfile
    payload=bundle.remote();out=ROOT/'results'/RUN
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:t.extractall(out,filter='data')
    print('Collected',len(payload),'bytes',flush=True)
