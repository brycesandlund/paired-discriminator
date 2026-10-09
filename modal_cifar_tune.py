"""Bounded optimizer/EMA screen and adaptive refinement on CIFAR."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('paired-discriminator-cifar-tuning')
data_volume=modal.Volume.from_name('paired-discriminator-cifar10-data')
gan_volume=modal.Volume.from_name('paired-discriminator-cifar10-runs')
flow_volume=modal.Volume.from_name('paired-discriminator-image-flow-runs')
run_volume=modal.Volume.from_name('paired-discriminator-cifar-tuning',create_if_missing=True)
volumes={'/data':data_volume,'/gan-runs':gan_volume,'/flow-runs':flow_volume,'/tuning':run_volume}
image=(modal.Image.debian_slim(python_version='3.12').uv_sync(uv_project_dir=str(ROOT),extras=['cifar'])
       .env({'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_HOME':'/data/torch','PYTHONPATH':'/root'})
       .add_local_dir(ROOT/'src/paired_discriminator','/root/paired_discriminator'))

@app.function(image=image,gpu='A10',cpu=4,memory=16384,volumes=volumes,timeout=10800,retries=1,max_containers=4,scaledown_window=2)
def job(spec:dict):
    import torch,numpy as np
    from paired_discriminator.cifar_tune import train,evaluate,configs
    from paired_discriminator.cifar import atomic_json
    run_volume.reload();gan_volume.reload();flow_volume.reload()
    root=Path('/tuning/cifar-tune-v1');root.mkdir(parents=True,exist_ok=True)
    method=spec['method'];seed=spec.get('seed',0);kind=spec['kind']
    if kind=='saved_ema':
        step=spec['step'];parent=Path(f'/flow-runs/image-flow-50k-v1/cifar10_seed0/velocity_{step:06d}.pt')
        r=evaluate(parent,'flow',0,'ema',root/f'saved_flow_ema_{step}',commit=run_volume.commit)
        return dict(spec=spec,evaluations={'ema':r})
    if kind=='verify':
        cfg=configs(method)[0]|{'batch_size':4,'name':'verify'}
        data=torch.from_numpy(np.load('/data/cifar10/train_uint8.npy')[:16].copy())
        a=train(method,0,cfg,4,root/'verify'/method/'full',data)
        train(method,0,cfg,2,root/'verify'/method/'split',data)
        b=train(method,0,cfg,4,root/'verify'/method/'split',data)
        for key in ('model','ema','rngs'):
            assert all(torch.equal(a[key][k],b[key][k]) for k in a[key])
        if method!='flow':assert all(torch.equal(a['discriminator'][k],b['discriminator'][k]) for k in a['discriminator'])
        def equal(a,b):
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
        for k in ['optimizer_g','optimizer_d']:assert equal(a[k],b[k])
        return dict(method=method,exact_cuda_resume=True)
    cfg=spec['config'];out=root/f"{spec.get('group','')}{method}_{cfg['name']}_seed{seed}"
    if spec.get('parent'):
        parent=Path(spec['parent'])
        assert parent.is_relative_to('/tuning/cifar-tune-v1'), 'Parent outside experiment'
    elif spec.get('fresh',False):
        parent=None
    elif method=='flow':parent=Path('/flow-runs/image-flow-50k-v1/cifar10_seed0/velocity_080000.pt')
    else:parent=Path(f'/gan-runs/cifar10-pilot-v1/{method}_seed0/latest.pt')
    if parent:
        old=torch.load(parent,map_location='cpu',weights_only=False)
        assert old['step']==spec.get('parent_step',80000 if method=='flow' else 50000),old['step']
    data=torch.from_numpy(np.load('/data/cifar10/train_uint8.npy'))
    results={}
    for step in spec.get('eval_steps',[spec['steps']]):
        existing=json.loads((out/'status.json').read_text())['step'] if (out/'status.json').exists() else 0
        if existing<=step:train(method,seed,cfg,step,out,data,parent,commit=run_volume.commit)
        for weights in ['model','ema']:
            r=evaluate(out/f'checkpoint_{step:06d}.pt',method,seed,weights,out/f'eval_{step:06d}_{weights}',noise_seed=spec.get('noise_seed',99174),commit=run_volume.commit)
            results[f'{step}_{weights}']=r
        status=json.loads((out/'status.json').read_text());status.update(phase='complete',target=spec['steps'])
        atomic_json(out/'status.json',status);run_volume.commit()
    return dict(spec=spec,evaluations=results)

@app.function(image=image,volumes={'/tuning':run_volume},timeout=21600)
def coordinate():
    from paired_discriminator.cifar_tune import configs
    from paired_discriminator.cifar import atomic_json
    root=Path('/tuning/cifar-tune-v1');root.mkdir(parents=True,exist_ok=True)
    checks=[job.spawn(dict(kind='verify',method=m)) for m in ['vanilla','paired','flow']]
    verification=[c.get() for c in checks];atomic_json(root/'verification.json',verification);run_volume.commit()
    specs=[dict(kind='saved_ema',method='flow',step=s) for s in [50000,80000,100000]]
    specs += [dict(kind='screen',method=m,config=c,steps=10000) for m in ['vanilla','paired','flow'] for c in configs(m)]
    calls=[job.spawn(s) for s in specs]
    atomic_json(root/'dispatch.json',dict(stage='screen',calls=[dict(spec=s,call_id=c.object_id) for s,c in zip(specs,calls)]));run_volume.commit()
    records=[];errors=[]
    for s,c in zip(specs,calls):
        try:records.append(c.get())
        except Exception as e:errors.append(dict(spec=s,error=str(e)))
        atomic_json(root/'screen.json',dict(records=records,errors=errors));run_volume.commit()
    if errors:raise RuntimeError(str(errors))
    selected=[]
    for m in ['vanilla','paired','flow']:
        ranked=[]
        for r in records:
            if r['spec']['kind']=='screen' and r['spec']['method']==m:
                best=min(r['evaluations'].values(),key=lambda x:x['train_metrics']['frechet_inception_distance'])
                ranked.append((best['train_metrics']['frechet_inception_distance'],r['spec']))
        winner=min(ranked,key=lambda x:x[0])[1]
        for cfg in [winner['config']]+([] if winner['config']['name']=='control' else [configs(m)[0]]):
            selected.append(dict(kind='refine',method=m,config=cfg,steps=30000,eval_steps=[20000,30000]))
    calls=[job.spawn(s) for s in selected]
    atomic_json(root/'refine_dispatch.json',dict(stage='refine',calls=[dict(spec=s,call_id=c.object_id) for s,c in zip(selected,calls)]));run_volume.commit()
    refined=[];errors=[]
    for s,c in zip(selected,calls):
        try:refined.append(c.get())
        except Exception as e:errors.append(dict(spec=s,error=str(e)))
        atomic_json(root/'refine.json',dict(records=refined,errors=errors));run_volume.commit()
    atomic_json(root/'stage_status.json',dict(phase='refinement_complete',next='Review results; confirm frozen candidates on fresh seeds.',errors=errors));run_volume.commit()
    return dict(screen_jobs=len(records),refined_jobs=len(refined),errors=errors)

@app.local_entrypoint()
def main():
    call=coordinate.spawn();out=ROOT/'results/cifar-tune-v1';out.mkdir(parents=True,exist_ok=True)
    (out/'launch.json').write_text(json.dumps(dict(call_id=call.object_id),indent=2)+'\n')
    print('COORDINATOR',call.object_id,flush=True);print(call.get(),flush=True)

@app.local_entrypoint()
def single(spec_path:str):
    call=job.spawn(json.loads(Path(spec_path).read_text()));print('JOB',call.object_id,flush=True);print(call.get(),flush=True)


@app.function(image=image,volumes={'/tuning':run_volume},timeout=21600)
def confirm_coordinate(recipes:dict):
    from paired_discriminator.cifar_tune import configs
    from paired_discriminator.cifar import atomic_json
    root=Path('/tuning/cifar-tune-v1');root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'confirmation_recipes.json',recipes);run_volume.commit()
    bases=[dict(kind='base',method=m,seed=seed,config=configs(m)[0],steps=80000 if m=='flow' else 50000,
                fresh=True,group='base_',noise_seed=99274) for m in ['vanilla','paired','flow'] for seed in [1,2]]
    calls=[job.spawn(s) for s in bases]
    atomic_json(root/'confirmation_base_dispatch.json',dict(calls=[dict(spec=s,call_id=c.object_id) for s,c in zip(bases,calls)]));run_volume.commit()
    results=[];errors=[]
    for s,c in zip(bases,calls):
        try:results.append(c.get())
        except Exception as e:errors.append(dict(spec=s,error=str(e)))
        atomic_json(root/'confirmation_bases.json',dict(records=results,errors=errors));run_volume.commit()
    if errors:raise RuntimeError(str(errors))
    specs=[]
    for method,recipe in recipes.items():
        for seed in [1,2]:
            base=80000 if method=='flow' else 50000
            if recipe.get('continuous'):
                specs.append(dict(kind='confirm',method=method,seed=seed,config=configs(method)[0],steps=recipe['total_steps'],
                    fresh=True,group='base_',noise_seed=99274))
            else:
                cfgs=[recipe['config']]
                if recipe['config']['name']!='control':cfgs.append(configs(method)[0])
                for cfg in cfgs:
                    specs.append(dict(kind='confirm',method=method,seed=seed,config=cfg,steps=recipe['branch_steps'],
                        group='confirm_',noise_seed=99274,parent=f'/tuning/cifar-tune-v1/base_{method}_control_seed{seed}/checkpoint_{base:06d}.pt',parent_step=base))
    calls=[job.spawn(s) for s in specs]
    atomic_json(root/'confirmation_dispatch.json',dict(calls=[dict(spec=s,call_id=c.object_id) for s,c in zip(specs,calls)]));run_volume.commit()
    results=[];errors=[]
    for s,c in zip(specs,calls):
        try:results.append(c.get())
        except Exception as e:errors.append(dict(spec=s,error=str(e)))
        atomic_json(root/'confirmation.json',dict(records=results,errors=errors));run_volume.commit()
    atomic_json(root/'confirmation_status.json',dict(phase='complete',errors=errors));run_volume.commit()
    return dict(jobs=len(results),errors=errors)

@app.local_entrypoint()
def confirm(recipes_path:str):
    recipes=json.loads(Path(recipes_path).read_text());call=confirm_coordinate.spawn(recipes)
    out=ROOT/'results/cifar-tune-v1';out.mkdir(parents=True,exist_ok=True)
    (out/'confirmation_launch.json').write_text(json.dumps(dict(call_id=call.object_id,recipes=recipes),indent=2)+'\n')
    print('CONFIRMATION',call.object_id,flush=True);print(call.get(),flush=True)
