"""CIFAR optimizer/EMA tuning, with isolated branches and reproducible evaluation."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import cifar, image_flow


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configs(method):
    if method=='flow':
        return [dict(name=n,lr_g=lr,lr_d=0.,decay=dec,r1=0.) for n,lr,dec in
                [('control',2e-4,False),('lr1e4',1e-4,False),('lr5e5',5e-5,False),('decay',2e-4,True)]]
    return [dict(name=n,lr_g=g,lr_d=d,decay=dec,r1=r1) for n,g,d,dec,r1 in [
        ('control',2e-4,2e-4,False,0.),('low_both',1e-4,1e-4,False,0.),
        ('slow_d',2e-4,1e-4,False,0.),('slow_g',1e-4,2e-4,False,0.),
        ('decay',2e-4,2e-4,True,0.),('r1',2e-4,2e-4,False,1.)]]


def lr_factor(step,cfg):
    return .1+.9*(1+math.cos(math.pi*min(step,30000)/30000))/2 if cfg.get('decay') else 1.


@torch.no_grad()
def update_ema(ema,model,decay=.999):
    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
    for a,b in zip(ema.buffers(),model.buffers()):
        if a.is_floating_point():a.lerp_(b,1-decay)
        else:a.copy_(b)


def models(method,seed):
    if method=='flow':
        return image_flow.make_model('cifar10',{'width':32},seed),None
    return cifar.models({'latent_dim':128,'width':64},method,seed)


def train(method,seed,cfg,steps,output,data,parent=None,device='cuda',commit=None):
    cifar.configure(device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source_hashes={n:digest(Path(__file__).with_name(n+'.py')) for n in ['cifar_tune','cifar','image_flow']}
    signature=dict(method=method,seed=seed,config=cfg,source_hashes=source_hashes,
        data_sha256=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),torch=torch.__version__,device=device,
        parent_sha256=digest(parent) if parent else None)
    g,d=models(method,seed);g=g.to(device);d=d.to(device) if d is not None else None
    ema=copy.deepcopy(g).requires_grad_(False)
    og=torch.optim.Adam(g.parameters(),lr=cfg['lr_g'],betas=(.9,.999) if method=='flow' else (.5,.999))
    od=torch.optim.Adam(d.parameters(),lr=cfg['lr_d'],betas=(.5,.999)) if d is not None else None
    rng=cifar.rngs_for(seed);rng['time']=torch.Generator().manual_seed(seed+(60000 if method=='flow' else 70000))
    latest=out/'latest.pt';start=0;seconds=0.;rows=[];base_step=0
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['signature']==signature,'Resume signature changed'
        g.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);og.load_state_dict(ck['optimizer_g'])
        if d is not None:d.load_state_dict(ck['discriminator']);od.load_state_dict(ck['optimizer_d'])
        for k,v in rng.items():v.set_state(ck['rngs'][k])
        start=ck['step'];seconds=ck['training_seconds'];rows=ck['rows'];base_step=ck['base_step']
    elif parent:
        ck=torch.load(parent,map_location='cpu',weights_only=False)
        g.load_state_dict(ck.get('model',ck.get('generator')))
        if d is not None:d.load_state_dict(ck['discriminator'])
        ema.load_state_dict(ck['ema'] if method=='flow' and isinstance(ck.get('ema'),dict) else g.state_dict())
        base_step=ck.get('base_step',0)+ck['step']
        # All screened branches reset Adam and sampling streams identically, including control.
    assert start<=steps
    data=data.to(device);batch=cfg.get('batch_size',128)
    provenance=dict(**signature,gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
        parameters_g=sum(p.numel() for p in g.parameters()),parameters_d=sum(p.numel() for p in d.parameters()) if d else 0,
        optimizer_reset_at_branch=bool(parent),base_step=base_step)
    cifar.atomic_json(out/'provenance.json',provenance)
    (out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    def draw(kind):
        ids=torch.randint(len(data),(batch,),generator=rng['real_'+kind]).to(device)
        return data[ids].float()/127.5-1
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    sync();began=time.perf_counter()
    for step in range(start+1,steps+1):
        g.train();f=lr_factor(step,cfg)
        if method=='flow' and base_step==0:f*=min(1.,step/500)
        og.param_groups[0]['lr']=cfg['lr_g']*f
        if method=='flow':
            real=draw('d');z=torch.randn(real.shape,generator=rng['noise_d']).to(device)
            t=torch.rand(batch,generator=rng['time']).to(device)
            xt,target=image_flow.interpolation(z,real,t)
            og.zero_grad(set_to_none=True);loss=F.mse_loss(g(xt,t),target);loss.backward();og.step();dl=None
        else:
            d.requires_grad_(True);od.param_groups[0]['lr']=cfg['lr_d']*f
            real=draw('d');z=torch.randn(batch,128,generator=rng['noise_d']).to(device)
            slots=(torch.randperm(batch,generator=rng['slots'])<batch//2).to(device)
            with torch.no_grad():fake=g(z)
            od.zero_grad(set_to_none=True);dl=cifar.loss_d(d,real,fake,method,slots)
            if cfg.get('r1',0)>0 and step%16==0:
                # Penalize all input-coordinate derivatives. Paired regularizer sees both slots.
                x=(cifar.pair_inputs(real,fake,slots) if method=='paired' else real).detach().requires_grad_(True)
                grad=torch.autograd.grad(d(x).sum(),x,create_graph=True)[0]
                dl=dl+.5*cfg['r1']*16*grad.flatten(1).square().sum(1).mean()
            dl.backward();od.step();d.requires_grad_(False)
            real=draw('g');z=torch.randn(batch,128,generator=rng['noise_g']).to(device)
            slots=(torch.randperm(batch,generator=rng['slots'])<batch//2).to(device)
            og.zero_grad(set_to_none=True);loss=cifar.loss_g(d,real,g(z),method,slots);loss.backward();og.step()
        update_ema(ema,g)
        if step%1000==0 or step==steps:
            sync();seconds+=time.perf_counter()-began
            assert torch.isfinite(loss) and (dl is None or torch.isfinite(dl))
            rows.append(dict(step=step,loss=float(loss.detach()),d_loss=float(dl.detach()) if dl is not None else None,training_seconds=seconds))
            ck=dict(signature=signature,step=step,base_step=base_step,model=g.state_dict(),ema=ema.state_dict(),
                discriminator=d.state_dict() if d else None,optimizer_g=og.state_dict(),optimizer_d=od.state_dict() if od else None,
                rngs={k:v.get_state() for k,v in rng.items()},training_seconds=seconds,rows=rows)
            torch.save(ck,out/'latest.pt.tmp');(out/'latest.pt.tmp').replace(latest)
            if step%10000==0 or step==steps:torch.save(ck,out/f'checkpoint_{step:06d}.pt')
            cifar.atomic_json(out/'status.json',dict(step=step,base_step=base_step,phase='training',training_seconds=seconds))
            cifar.atomic_json(out/'training.json',rows)
            print(method,seed,cfg['name'],step,seconds,flush=True)
            if commit:commit()
            sync();began=time.perf_counter()
    return ck


def evaluate(checkpoint,method,seed,weights,output,noise_seed=99174,device='cuda',commit=None):
    from torchvision.utils import save_image
    from .cifar_deficit import load_classifier,WEIGHTS
    from .cifar_class_representation import predict,summarize
    from .integrity_eval import extractor,extract,metrics
    cifar.configure(device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    signature=dict(checkpoint_sha256=digest(checkpoint),source_sha256=digest(__file__),method=method,seed=seed,
                   weights=weights,noise_seed=noise_seed,samples=10000,solver_steps=64 if method=='flow' else None)
    if (out/'metrics.json').exists():
        r=json.loads((out/'metrics.json').read_text());assert r['signature']==signature;return r
    began=time.perf_counter();ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    g,_=models(method,seed);g.load_state_dict(ck[weights] if weights=='ema' else ck.get('model',ck.get('generator')));g.to(device).eval()
    rng=torch.Generator().manual_seed(noise_seed);images=[];first=None
    sample_start=time.perf_counter()
    with torch.no_grad():
        for start in range(0,10000,128):
            n=min(128,10000-start);shape=(n,3,32,32) if method=='flow' else (n,128)
            z=torch.randn(shape,generator=rng).to(device)
            x=image_flow.sample(g,z,64) if method=='flow' else g(z)
            if first is None:first=x.cpu();first_z=z
            images.append(image_flow.to_uint8(x).cpu())
        replay=image_flow.sample(g,first_z,64) if method=='flow' else g(first_z)
        assert torch.equal(first,replay.cpu())
    fake=torch.cat(images);sampling_seconds=time.perf_counter()-sample_start
    save_image(fake[:64].float()/255,out/'samples.png',nrow=8);del g,replay
    result=dict(signature=signature,step=ck['step'],base_step=ck.get('base_step',0),first_batch_replay_exact=True,
        sampling_seconds=sampling_seconds,classifiers={},classifier_weight_hashes=WEIGHTS,
        images_sha256=hashlib.sha256(fake.numpy().tobytes()).hexdigest())
    labels=[]
    for name in WEIGHTS:
        c=load_classifier(name,'/data/class-representation-weights',device);p=predict(c,fake,device)
        result['classifiers'][name]=summarize(p);labels.append(p.argmax(1));del c
    result['classifier_agreement']=float((labels[0]==labels[1]).mean())
    ext=extractor(device);features=extract(fake.numpy(),ext,device);del ext
    cache=Path('/data/cifar10/integrity-v1')
    test=torch.load(cache/'test_features.pt',map_location='cpu',weights_only=True)
    train=torch.load(cache/'train_features.pt',map_location='cpu',weights_only=True)
    ids=np.load(cache/'reference_ids.npz')['train_metrics']
    result['train_metrics']=metrics(features,train[ids],device)
    result['heldout_metrics']=metrics(features,test,device)
    result['reference_manifest']=json.loads((cache/'manifest.json').read_text())
    result['evaluation_seconds']=time.perf_counter()-began
    result['selection_metric']='train-reference FID; test-reference metrics are descriptive, not untouched validation'
    (out/'evaluation_source.py').write_bytes(Path(__file__).read_bytes());cifar.atomic_json(out/'metrics.json',result)
    if commit:commit()
    return result
