"""Auditable short flow-matching ablations; oracle supervision is diagnostic only."""
import argparse,copy,json,math,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from .grid_experiment import ROOT,sample_real,snapshot
from .grid10_flow import sample
from .grid10_flow_oracle import Oracle
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

class Field(nn.Module):
    def __init__(self, spec, scale):
        super().__init__();self.spec=spec;self.scale=scale
        self.register_buffer('freq',torch.logspace(-1,4,10,base=2))
        dims=3+(40+20 if spec.get('fourier') else 0)
        width=spec.get('width',128);depth=spec.get('depth',2)
        layers=[]
        for i in range(depth):layers += [nn.Linear(dims if i==0 else width,width),nn.SiLU()]
        layers += [nn.Linear(width,2)];self.net=nn.Sequential(*layers)
    def forward(self,x,t):
        # Scaling chosen from a separate sample-only pilot, never from mode labels.
        residual=self.spec.get('precondition',False)
        if residual:
            var=(1-t).square()+(t*self.scale).square()
            coords=x/var.sqrt()
            base=((t*self.scale**2-(1-t))/var)*x
        else:
            coords=x/self.scale if self.spec.get('normalize') else x
            base=0
        features=[coords,t]
        if self.spec.get('fourier'):
            a=2*math.pi*coords[:,:,None]*self.freq
            b=2*math.pi*t*self.freq
            features += [a.sin().flatten(1),a.cos().flatten(1),b.sin(),b.cos()]
        return base+self.net(torch.cat(features,1))

def train(args):
    root,spec,seed,steps=args;root=Path(root);p=root/f"{spec['name']}_seed{seed}";p.mkdir()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    c=json.loads((root/'config.json').read_text());scale=c['pilot_scale'];model=Field(spec,scale)
    ema=copy.deepcopy(model);beta=spec.get('ema',.999)
    opt=torch.optim.Adam(model.parameters(),lr=spec.get('lr',.001),betas=(.9,.999))
    rngs={k:torch.Generator().manual_seed(seed+offset) for k,offset in [('real',20000),('noise',40000),('time',50000)]}
    z=torch.randn(c['eval_samples'],2,generator=torch.Generator().manual_seed(seed+70000))
    oracle=Oracle(c);target_grid_value=target_grid(c,EDGES);rows=[];train_s=0;start=time.perf_counter()
    ckpts=sorted(set([steps//2,steps]));batch=spec.get('batch',256)
    for step in range(1,steps+1):
        tick=time.perf_counter()
        x1=sample_real(c,batch,rngs['real']);x0=torch.randn(batch,2,generator=rngs['noise']);t=torch.rand(batch,1,generator=rngs['time'])
        if spec.get('ot'):
            _,perm=linear_sum_assignment(torch.cdist(x0,x1).square().numpy());x1=x1[perm]
        if spec.get('late'):
            t=t.sqrt() # deliberate time reweighting, not an unbiased uniform-time estimator
        xt=(1-t)*x0+t*x1
        with torch.no_grad():v=oracle(xt,t) if spec.get('oracle') else x1-x0
        loss=(model(xt,t)-v).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-beta)
        train_s+=time.perf_counter()-tick
        if step in ckpts:
            assert torch.isfinite(loss)
            saved=dict(spec=spec,config=c,scale=scale,step=step,seed=seed,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),rng={k:r.get_state() for k,r in rngs.items()})
            torch.save(saved,p/f'checkpoint_{step}.pt')
            for typ,net in [('raw',model),('ema',ema)]:
                points=sample(net,z,128).numpy();assert np.isfinite(points).all();m,_=metrics(points,c,target_grid_value)
                np.save(p/f'samples_{step}_{typ}.npy',points)
                rows.append(dict(step=step,seed=seed,name=spec['name'],variant=typ,train_seconds=train_s,**m))
            (p/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(spec['name'],seed,step,[(r['variant'],round(r['mode_tv'],4),round(r['valid_fraction'],3)) for r in rows[-2:]],flush=True)
    (p/'metadata.json').write_text(json.dumps(dict(steps=steps,seed=seed,parameters=sum(v.numel() for v in model.parameters()),training_seconds=train_s,wall_seconds=time.perf_counter()-start,real_samples=steps*batch,oracle_supervised=spec.get('oracle',False)),indent=2)+'\n')
    return rows[-2:]

SPECS=[
 dict(name='adam',depth=2),
 dict(name='deep',depth=4),
 dict(name='normalized',depth=3,normalize=True),
 dict(name='fourier',depth=3,normalize=True,fourier=True),
 dict(name='precond',depth=3,precondition=True),
 dict(name='precond_fourier',depth=3,precondition=True,fourier=True),
 dict(name='oracle_small',depth=2,oracle=True),
 dict(name='oracle_fourier',depth=3,precondition=True,fourier=True,oracle=True),
]
def launch(root_name,specs,steps=20000,seeds=(0,),workers=8,eval_samples=4000):
    root=ROOT/'results'/root_name
    c=json.loads((ROOT/'configs/grid10-1200k.json').read_text());c.update(steps=steps,eval_samples=eval_samples)
    pilot=sample_real(c,100000,torch.Generator().manual_seed(18373));c['pilot_scale']=float(pilot.std(unbiased=False));c['pilot_samples']=100000
    snapshot(root,c);(root/'specs.json').write_text(json.dumps(specs,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(train,(str(root),spec,seed,steps)) for spec in specs for seed in seeds]):print('DONE',f.result(),flush=True)
if __name__=='__main__':launch('flow-ablation-r1-v1',SPECS)
