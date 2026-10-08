"""Short GAN ablations: unchanged BCE objectives; features, Adam, D-only sampler."""
import copy,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch import nn
from .grid_experiment import ROOT,sample_real,snapshot
from .grid_pacgan import models as old_models, discriminator_loss,generator_loss
from .grid10_strong import accepted_mass,deficit_weights,sample_weighted
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

class FeatureD(nn.Module):
    def __init__(self,spec,scale):
        super().__init__();self.spec=spec;self.scale=scale
        self.register_buffer('freq',torch.logspace(-1,4,10,base=2));self.register_buffer('gate',torch.ones(10))
        dims=2 if spec.get('method','paired')=='vanilla' else 4
        n=dims*(21 if spec.get('fourier') else 1);width=spec.get('d_width',128)
        act=nn.SiLU if spec.get('silu') else lambda:nn.LeakyReLU(.2)
        self.net=nn.Sequential(nn.Linear(n,width),act(),nn.Linear(width,width),act(),nn.Linear(width,1))
    def forward(self,x):
        y=x/self.scale if self.spec.get('normalized') or self.spec.get('fourier') else x
        if self.spec.get('fourier'):
            a=2*math.pi*y[:,:,None]*self.freq
            y=torch.cat([y,(a.sin()*self.gate).flatten(1),(a.cos()*self.gate).flatten(1)],1)
        return self.net(y)
    def progress(self,fraction):
        if self.spec.get('anneal'):
            self.gate.copy_((fraction*2*10-torch.arange(10,dtype=self.gate.dtype)).clamp(0,1))

class FeatureG(nn.Module):
    def __init__(self,spec):
        super().__init__();latent=spec.get('latent',16)
        self.register_buffer('freq',torch.logspace(-1,4,10,base=2))
        self.net=nn.Sequential(nn.Linear(latent+40,128),nn.LeakyReLU(.2),nn.Linear(128,128),nn.LeakyReLU(.2),nn.Linear(128,2))
    def forward(self,z):
        a=2*math.pi*z[:,:2,None]*self.freq
        return self.net(torch.cat([z,a.sin().flatten(1),a.cos().flatten(1)],1))

class ResidualG(nn.Module):
    def __init__(self,base,scale):
        super().__init__();self.net=base;self.scale=scale
    def forward(self,z):return self.scale*(z[:,:2]+self.net(z))

def make_models(c,spec,seed):
    g,_=old_models(dict(c,latent_dim=spec.get('latent',16)),spec.get('method','paired'),seed)
    if spec.get('g_fourier'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);g=FeatureG(spec)
    if spec.get('residual_g'):g=ResidualG(g,c['pilot_scale'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+10000);d=FeatureD(spec,c['pilot_scale'])
    return g,d

def train(args):
    root,spec,seed,steps=args;root=Path(root);p=root/f"{spec['name']}_seed{seed}";p.mkdir()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c=json.loads((root/'config.json').read_text());g,d=make_models(c,spec,seed);g_ema=copy.deepcopy(g)
    method=spec.get('method','paired');batch=spec.get('batch',256);latent=spec.get('latent',16)
    og=torch.optim.Adam(g.parameters(),lr=spec.get('lr_g',.0002),betas=tuple(spec.get('betas',[.5,.999])))
    od=torch.optim.Adam(d.parameters(),lr=spec.get('lr_d',spec.get('lr_g',.0002)),betas=tuple(spec.get('betas',[.5,.999])))
    rngs={k:torch.Generator().manual_seed(seed+off) for k,off in [('real_d',20000),('real_g',30000),('noise_d',40000),('noise_g',50000),('slots',60000)]}
    z=torch.randn(c['eval_samples'],latent,generator=torch.Generator().manual_seed(seed+70000));target=target_grid(c,EDGES)
    mass=torch.full((100,),.01,dtype=torch.float64);rows=[];train_s=0.;started=time.perf_counter()
    for step in range(1,steps+1):
        tick=time.perf_counter();fraction=(step-1)/steps;d.progress(fraction)
        if spec.get('cosine'):
            for opt,lr in [(og,spec.get('lr_g',.0002)),(od,spec.get('lr_d',spec.get('lr_g',.0002)))]:
                decay_fraction=max(0.,(fraction-spec.get('cosine_after',0))/(1-spec.get('cosine_after',0)))
                opt.param_groups[0]['lr']=lr*(.1+.9*.5*(1+math.cos(math.pi*decay_fraction)))
        alpha=spec.get('alpha',.9)
        if spec.get('ramp_alpha'):alpha*=min(1,fraction*2)
        weights=deficit_weights(mass,1-alpha,spec.get('power',2))
        real=sample_weighted(c,batch,rngs['real_d'],weights) if alpha>0 else sample_real(c,batch,rngs['real_d'])
        noise=torch.randn(batch,latent,generator=rngs['noise_d']);slots=torch.randperm(batch,generator=rngs['slots'])<batch//2
        d.requires_grad_(True);od.zero_grad(set_to_none=True)
        with torch.no_grad():fake=g(noise);observed=accepted_mass(fake,c)
        ld=discriminator_loss(d,real,fake,method,slots);ld.backward();od.step()
        real=sample_real(c,batch,rngs['real_g']);noise=torch.randn(batch,latent,generator=rngs['noise_g']);slots=torch.randperm(batch,generator=rngs['slots'])<batch//2
        d.requires_grad_(False);og.zero_grad(set_to_none=True);lg=generator_loss(d,real,g(noise),method,slots);lg.backward();og.step()
        with torch.no_grad():
            for a,b in zip(g_ema.parameters(),g.parameters()):a.lerp_(b,.001)
        decay=spec.get('mass_decay',.99);mass=decay*mass+(1-decay)*observed
        train_s+=time.perf_counter()-tick
        if step in (steps//2,steps):
            assert torch.isfinite(ld+lg)
            torch.save(dict(config=c,spec=spec,seed=seed,step=step,generator=g.state_dict(),discriminator=d.state_dict(),ema=g_ema.state_dict(),optimizer_g=og.state_dict(),optimizer_d=od.state_dict(),mass=mass,rng={k:r.get_state() for k,r in rngs.items()}),p/f'checkpoint_{step}.pt')
            for variant,net in [('raw',g),('ema',g_ema)]:
                with torch.no_grad():points=net(z).numpy()
                assert np.isfinite(points).all();m,_=metrics(points,c,target);np.save(p/f'samples_{step}_{variant}.npy',points)
                rows.append(dict(name=spec['name'],seed=seed,step=step,variant=variant,training_seconds=train_s,**m))
            (p/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n');print(spec['name'],seed,step,[(r['variant'],round(r['mode_tv'],4),r['coverage_half_target']) for r in rows[-2:]],flush=True)
    (p/'metadata.json').write_text(json.dumps(dict(steps=steps,seed=seed,parameters_g=sum(v.numel() for v in g.parameters()),parameters_d=sum(v.numel() for v in d.parameters()),training_seconds=train_s,wall_seconds=time.perf_counter()-started,d_real_samples=steps*batch,g_real_references=steps*batch if method=='paired' else 0),indent=2)+'\n')
    return rows[-2:]

SPECS=[
 dict(name='control'),
 dict(name='lr1e3',lr_g=.001),
 dict(name='adam09',lr_g=.001,betas=[.9,.999]),
 dict(name='normalized',normalized=True),
 dict(name='fourier',fourier=True),
 dict(name='fourier_fast',fourier=True,lr_g=.001),
 dict(name='fourier_beta0',fourier=True,lr_g=.001,betas=[0.,.99]),
 dict(name='fourier_anneal',fourier=True,lr_g=.001,anneal=True),
 dict(name='fourier_weak',fourier=True,lr_g=.001,alpha=.5,power=1),
 dict(name='fourier_uniform',fourier=True,lr_g=.001,alpha=0,power=1),
]
def launch(name,specs,steps=30000,seeds=(0,),workers=8,eval_samples=4000):
    root=ROOT/'results'/name
    c=json.loads((ROOT/'configs/grid10-strong-1200k.json').read_text());c.update(steps=steps,eval_samples=eval_samples)
    pilot=sample_real(c,100000,torch.Generator().manual_seed(18373));c['pilot_scale']=float(pilot.std(unbiased=False))
    snapshot(root,c);(root/'specs.json').write_text(json.dumps(specs,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(train,(str(root),spec,seed,steps)) for spec in specs for seed in spec.get("seeds",seeds)]):print('DONE',f.result(),flush=True)
if __name__=='__main__':launch('gan-ablation-r1-v1',SPECS)
