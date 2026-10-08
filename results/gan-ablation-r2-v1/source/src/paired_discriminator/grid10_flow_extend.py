"""Independent-coupling straight-line conditional flow matching on the 10x10 grid."""
from pathlib import Path
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch import nn
from .grid_experiment import ROOT,sample_real,snapshot
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,x,t):
        return self.net(torch.cat([x,t],dim=1))

@torch.no_grad()
def sample(model,z,steps):
    x=z.clone();h=1/steps
    for i in range(steps):
        t=torch.full((len(x),1),i*h)
        v=model(x,t)
        x=x+h*model(x+h*v/2,t+h/2)
    return x

def job(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    root=ROOT/'results/grid10-flow-1200k-v1';p=root/f'seed{seed}';p.mkdir()
    c=json.loads((root/'config.json').read_text());model=Velocity()
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']))
    real_rng=torch.Generator().manual_seed(seed+20000);noise_rng=torch.Generator().manual_seed(seed+40000);time_rng=torch.Generator().manual_seed(seed+50000)
    z=torch.randn(10000,2,generator=torch.Generator().manual_seed(seed+70000));target=target_grid(c,EDGES)
    source=ROOT/f'results/grid10-flow-v1/seed{seed}/checkpoint_50000.pt'
    state=torch.load(source,weights_only=False)
    assert state['step']==50000 and state['seed']==seed
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    for key,rng in [('real',real_rng),('noise',noise_rng),('time',time_rng)]:rng.set_state(state['rng_states'][key])
    prior=json.loads((source.parent/'metadata.json').read_text())
    assert np.array_equal(sample(model,z,128).numpy(),np.load(source.parent/'samples_50000_solver128.npy'))
    (p/'resume.json').write_text(json.dumps({'source':str(source),'step':50000,'boundary_replay':True})+'\n')
    rows=[];training_seconds=prior['training_seconds'];started=time.perf_counter()
    for step in range(50001,1200001):
        tick=time.perf_counter();x1=sample_real(c,256,real_rng);x0=torch.randn(256,2,generator=noise_rng);t=torch.rand(256,1,generator=time_rng)
        xt=(1-t)*x0+t*x1;loss=(model(xt,t)-(x1-x0)).square().mean()
        opt.zero_grad();loss.backward();opt.step();training_seconds+=time.perf_counter()-tick
        if step in (150000,300000,600000,1200000):
            assert torch.isfinite(loss)
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,seed=seed,config=c,rng_states={k:r.get_state() for k,r in [('real',real_rng),('noise',noise_rng),('time',time_rng)]}),p/f'checkpoint_{step}.pt')
            # Midpoint integration: 2 network evaluations per step.
            for steps in (64,128,256):
                tick=time.perf_counter();points=sample(model,z,steps).numpy();seconds=time.perf_counter()-tick
                assert points.shape==(10000,2) and np.isfinite(points).all()
                m,_=metrics(points,c,target);np.save(p/f'samples_{step}_solver{steps}.npy',points)
                rows.append(dict(seed=seed,step=step,solver_steps=steps,nfe=2*steps,training_seconds=training_seconds,sampling_seconds=seconds,real_samples=step*256,**m))
            (p/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(seed,step,'mode TV',rows[-2]['mode_tv'],flush=True)
    (p/'metadata.json').write_text(json.dumps(dict(seed=seed,steps=1200000,parameters=sum(v.numel() for v in model.parameters()),training_seconds=training_seconds,wall_seconds=time.perf_counter()-started),indent=2)+'\n')
    return seed

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .grid_experiment import centers
    root=ROOT/'results/grid10-flow-1200k-v1';c=json.loads((root/'config.json').read_text());rows=[]
    for seed in range(5):rows+=json.loads((root/f'seed{seed}/metrics.json').read_text())
    assert len(rows)==60
    (root/'all_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Straight-line flow matching on the 10×10 grid','','Five seeds; independent standard-normal noise and uniform real points. Time-uniform velocity MSE; two 128-wide SiLU layers; batch 256; Adam lr .0002, betas (.5,.999). No labels or deficit sampling. Explicit midpoint ODE solver.','','| Updates | Solver steps (NFE=2×steps) | Mode TV | Fine TV | Valid mass | Half-target coverage |','|---|---:|---:|---:|---:|---:|']
    for step in (150000,300000,600000,1200000):
        for solver in (64,128,256):
            rr=[r for r in rows if r['step']==step and r['solver_steps']==solver]
            lines.append(f'| {step} | {solver} | '+' | '.join(f'{np.mean([r[k] for r in rr]):.4f}' for k in ('mode_tv','spatial_tv','valid_fraction','coverage_half_target'))+' |')
    fig,axes=plt.subplots(1,4,figsize=(20,5));means=centers(c).numpy()
    for ax,step in zip(axes,(150000,300000,600000,1200000)):
        x=np.load(root/f'seed0/samples_{step}_solver128.npy');ax.scatter(x[:,0],x[:,1],s=2,alpha=.25);ax.scatter(means[:,0],means[:,1],s=8,c='red');ax.set(title=f'{step//1000}k updates',xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal')
    fig.suptitle('Flow matching • fixed seed 0 • 10,000 samples • midpoint 128 steps');fig.tight_layout();fig.savefig(root/'samples.png',dpi=170);plt.close(fig)
    lines+=['','![Samples](samples.png)','','Training time and sampling time per seed/checkpoint are recorded in all_metrics.json. Equal update counts do not imply equal GAN compute budgets. Solver comparisons reuse identical noise.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    root=ROOT/'results/grid10-flow-1200k-v1'
    c=json.loads((ROOT/'configs/grid10-1200k.json').read_text());c.update(steps=1200000,latent_dim=2,method='flow_matching',reference_phase='none',eval_steps=[150000,300000,600000,1200000],solver_steps=[64,128,256]);snapshot(root,c)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(job,s) for s in range(5)]):print('COMPLETED',f.result(),flush=True)
    report()
if __name__=='__main__':main()
