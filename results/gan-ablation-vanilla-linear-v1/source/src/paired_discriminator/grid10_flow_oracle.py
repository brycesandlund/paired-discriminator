"""Exact marginal velocity for independent Gaussian noise to Gaussian-mixture data."""
import json,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid10_flow import Velocity,sample
from .grid_experiment import ROOT,centers
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

class Oracle(torch.nn.Module):
    def __init__(self,c):
        super().__init__();self.register_buffer('means',centers(c));self.sigma2=c['sigma']**2
    def forward(self,x,t):
        # x_t | k ~ N(t mu_k, s² I), s²=(1-t)²+t² sigma².
        variance=(1-t).square()+t.square()*self.sigma2
        delta=x[:,None,:]-t[:,None,:]*self.means[None,:,:]
        weights=torch.softmax(-delta.square().sum(-1)/(2*variance),dim=1)
        mean=weights@self.means
        slope=(t*self.sigma2-(1-t))/variance
        return mean+slope*(x-t*mean)

def main():
    torch.set_num_threads(1);out=ROOT/'results/grid10-flow-oracle-v1';out.mkdir(exist_ok=True)
    c=json.loads((ROOT/'results/grid10-flow-1200k-v1/config.json').read_text());oracle=Oracle(c);target=target_grid(c,EDGES)
    # Endpoint identities: E[x1-x0|x0=x]=-x; E[x1-x0|x1=x]=x.
    x=torch.randn(32,2);assert torch.allclose(oracle(x,torch.zeros(32,1)),-x,atol=1e-5)
    assert torch.allclose(oracle(x,torch.ones(32,1)),x,atol=1e-5)
    state=torch.load(ROOT/'results/grid10-flow-1200k-v1/seed0/checkpoint_1200000.pt',weights_only=False)
    learned=Velocity();learned.load_state_dict(state['model'])
    z=torch.randn(10000,2,generator=torch.Generator().manual_seed(70000));results=[];outputs={}
    for steps in (64,128,256):
        tick=time.perf_counter()
        # Chunk to bound mixture posterior memory.
        points=torch.cat([sample(oracle,b,steps) for b in z.split(1000)]).numpy()
        metric,_=metrics(points,c,target);results.append(dict(solver_steps=steps,seconds=time.perf_counter()-tick,**metric));outputs[steps]=points
        np.save(out/f'oracle_samples_solver{steps}.npy',points)
    rng=torch.Generator().manual_seed(88221);comparison=[]
    with torch.no_grad():
        x0=torch.randn(20000,2,generator=rng);idx=torch.randint(100,(20000,),generator=rng);x1=centers(c)[idx]+.1*torch.randn(20000,2,generator=rng)
        for timepoint in (0.,.25,.5,.75,.9,.95,.98,.99,1.):
            t=torch.full((len(x0),1),timepoint);xt=(1-t)*x0+t*x1
            exact=oracle(xt,t);pred=learned(xt,t)
            comparison.append(dict(t=timepoint,velocity_rmse=float((pred-exact).square().mean().sqrt()),oracle_rms=float(exact.square().mean().sqrt())))
    (out/'metrics.json').write_text(json.dumps(dict(oracle_sampling=results,learned_field_error_seed0=comparison),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,5));means=centers(c).numpy()
    rngnp=np.random.default_rng(70000);real=means[rngnp.integers(100,size=10000)]+rngnp.normal(0,.1,(10000,2))
    original=np.load(ROOT/'results/grid10-flow-1200k-v1/seed0/samples_1200000_solver128.npy')
    for ax,title,points in zip(axes,['Learned flow • 1.2M','Analytic flow • 128 steps','True mixture reference'],[original,outputs[128],real]):
        ax.scatter(points[:,0],points[:,1],s=2,alpha=.25);ax.scatter(means[:,0],means[:,1],s=6,c='red');ax.set(title=title,xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Exact flow velocity diagnostic','','For x0 ~ N(0,I), x1 ~ uniform mixture N(mu_k,sigma² I), independent, and x_t=(1-t)x0+t*x1:', '', 's²=(1-t)²+t² sigma²; w_k(x,t)=softmax_k(-||x-t mu_k||²/(2s²)); mu_bar=sum_k w_k mu_k.', '', 'v(x,t)=mu_bar + [t sigma²-(1-t)]/s² × (x-t mu_bar).', '', 'This is E[x1-x0 | x_t=x], the exact regression target averaged over all compatible pairs. It uses the known target mixture: a diagnostic oracle, not a learned baseline. Endpoint identities v(x,0)=-x and v(x,1)=x checked.', '', '![Samples](comparison.png)', '', '| Solver steps | Mode TV | Fine TV | Valid mass | Half-target coverage |','|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['solver_steps']} | {r['mode_tv']:.4f} | {r['spatial_tv']:.4f} | {r['valid_fraction']:.4f} | {r['coverage_half_target']} |")
    lines+=['','Learned-field error uses 20k points drawn from the true interpolation marginal at each time, seed-0 learned checkpoint. Full values in metrics.json.','', '| t | Velocity RMSE per coordinate | Oracle velocity RMS |','|---|---:|---:|']
    for r in comparison:lines.append(f"| {r['t']} | {r['velocity_rmse']:.4f} | {r['oracle_rms']:.4f} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(sampling=results,field_error=comparison),indent=2))
if __name__=='__main__':main()
