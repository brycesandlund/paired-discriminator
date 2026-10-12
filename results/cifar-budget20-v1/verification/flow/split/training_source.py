"""Experiment40: existing recipes under a measured A10 training-time cap.

No architecture/objective search. Evaluations and checkpoint I/O are timed
separately. Budget checks synchronize CUDA every ten updates, giving a small
block-sized overshoot; interrupted jobs restore cumulative training time.
"""
import copy
import hashlib
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import cifar, image_flow
from .cifar_tune import digest, models, update_ema, lr_factor


def config(method):
    return dict(name='budget20',lr_g=2e-4,lr_d=0. if method=='flow' else 2e-4,
                decay=False,r1=0.,reset_step=50000 if method=='paired' else None)


def train(method,seed,cfg,steps,output,data,device='cuda',commit=None,budget_seconds=1200.):
    assert method in ('paired','flow') and budget_seconds>0
    cifar.configure(device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source_hashes={n:digest(Path(__file__).with_name(n+'.py')) for n in ['cifar_budget','cifar_tune','cifar','image_flow']}
    signature=dict(method=method,seed=seed,config=cfg,source_hashes=source_hashes,
        data_sha256=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),torch=torch.__version__,device=device,
        budget_seconds=budget_seconds)
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
    assert start<=steps
    if start and seconds>=budget_seconds:
        return ck
    data=data.to(device);batch=cfg.get('batch_size',128)
    provenance=dict(**signature,gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
        parameters_g=sum(p.numel() for p in g.parameters()),parameters_d=sum(p.numel() for p in d.parameters()) if d else 0,
        optimizer_reset_step=cfg.get('reset_step'),base_step=base_step)
    cifar.atomic_json(out/'provenance.json',provenance)
    (out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    def draw(kind):
        ids=torch.randint(len(data),(batch,),generator=rng['real_'+kind]).to(device)
        return data[ids].float()/127.5-1
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    sync();began=time.perf_counter()
    for step in range(start+1,steps+1):
        if method=='paired' and step==cfg.get('reset_step',-1)+1:
            # Reproduce experiment37's chosen branch: raw G initializes EMA;
            # both Adam optimizers and explicit sampling streams reset.
            ema.load_state_dict(g.state_dict())
            og=torch.optim.Adam(g.parameters(),lr=cfg['lr_g'],betas=(.5,.999))
            od=torch.optim.Adam(d.parameters(),lr=cfg['lr_d'],betas=(.5,.999))
            rng=cifar.rngs_for(seed);rng['time']=torch.Generator().manual_seed(seed+70000)
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
        exhausted=False
        if step%10==0 or step==steps:
            sync()
            exhausted=seconds+time.perf_counter()-began>=budget_seconds
        if step%1000==0 or step==steps or exhausted:
            sync();seconds+=time.perf_counter()-began
            assert torch.isfinite(loss) and (dl is None or torch.isfinite(dl))
            rows.append(dict(step=step,loss=float(loss.detach()),d_loss=float(dl.detach()) if dl is not None else None,training_seconds=seconds))
            ck=dict(signature=signature,step=step,base_step=base_step,model=g.state_dict(),ema=ema.state_dict(),
                discriminator=d.state_dict() if d else None,optimizer_g=og.state_dict(),optimizer_d=od.state_dict() if od else None,
                rngs={k:v.get_state() for k,v in rng.items()},training_seconds=seconds,rows=rows)
            torch.save(ck,out/'latest.pt.tmp');(out/'latest.pt.tmp').replace(latest)
            if step%10000==0 or step==steps or exhausted:torch.save(ck,out/f'checkpoint_{step:06d}.pt')
            cifar.atomic_json(out/'status.json',dict(step=step,base_step=base_step,phase='training_budget_complete' if exhausted else 'training',training_seconds=seconds,budget_seconds=budget_seconds))
            cifar.atomic_json(out/'training.json',rows)
            print(method,seed,cfg['name'],step,seconds,flush=True)
            if commit:commit()
            if exhausted:break
            sync();began=time.perf_counter()
    return ck

