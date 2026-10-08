"""Frozen-D coordinate gradients and a copied-checkpoint Adam G step; seed 0."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid10_continue import ROOT, models, centers, sample_real, generator_loss, accepted_mass


def main():
    torch.set_num_threads(1)
    out=ROOT/'results/grid10-gradient-diagnostic-v1';out.mkdir(exist_ok=True)
    methods=['vanilla','paired','vanilla_deficit','paired_deficit']
    fig,axes=plt.subplots(2,4,figsize=(20,10));summary=[]
    for col,method in enumerate(methods):
        state=torch.load(ROOT/f'results/grid10-1200k-v1/{method}_seed0/checkpoint.pt',weights_only=False)
        c=state['config'];g,d=models(c,method,0)
        g.load_state_dict(state['generator']);d.load_state_dict(state['discriminator']);d.requires_grad_(False)
        z=torch.randn(10000,c['latent_dim'],generator=torch.Generator().manual_seed(70000))
        with torch.no_grad():before=g(z).numpy()
        mass=accepted_mass(torch.from_numpy(before),c).numpy();means=centers(c).numpy()
        # Common uniform real references, both slots exactly averaged. No D update.
        refs=sample_real(c,128,torch.Generator().manual_seed(91001))
        axis=np.linspace(-7.5,7.5,45);xx,yy=np.meshgrid(axis,axis)
        points=np.column_stack([xx.ravel(),yy.ravel()]).astype('float32');vectors=[]
        for chunk in np.array_split(points,45):
            x=torch.tensor(chunk,requires_grad=True)
            if method.startswith('paired'):
                fake=x[:,None,:].expand(-1,len(refs),-1).reshape(-1,2)
                real=refs.repeat(len(x),1)
                loss=sum(generator_loss(d,real,fake,method,torch.full((len(fake),),slot,dtype=torch.bool)) for slot in [False,True])/2
            else:loss=generator_loss(d,x,x,method,torch.zeros(len(x),dtype=torch.bool))
            vectors.append((-torch.autograd.grad(loss,x)[0]*len(x)).numpy())
        vectors=np.concatenate(vectors);norm=np.linalg.norm(vectors,axis=1)
        # One ordinary-size G-only step using restored Adam moments, frozen D.
        opt=torch.optim.Adam(g.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']))
        opt.load_state_dict(state['optimizer_g'])
        real=sample_real(c,256,torch.Generator().manual_seed(92001))
        noise=torch.randn(256,c['latent_dim'],generator=torch.Generator().manual_seed(92002))
        slots=torch.randperm(256,generator=torch.Generator().manual_seed(92003))<128
        opt.zero_grad();loss=generator_loss(d,real,g(noise),method,slots);loss.backward();opt.step()
        with torch.no_grad():after=g(z).numpy()
        movement=after-before
        for row in range(2):
            ax=axes[row,col];ax.scatter(before[:,0],before[:,1],s=1,color='gray',alpha=.15)
            ax.scatter(means[:,0],means[:,1],s=15,marker='+',color='black')
            weak=mass<.0008;ax.scatter(means[weak,0],means[weak,1],s=95,facecolors='none',edgecolors='red',linewidths=1.3)
            ax.set(xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal',title=method.replace('_',' '))
        q=axes[0,col].quiver(points[:,0],points[:,1],vectors[:,0]/np.maximum(norm,1e-12),vectors[:,1]/np.maximum(norm,1e-12),np.log10(np.maximum(norm,1e-8)),angles='xy',scale_units='xy',scale=4,cmap='viridis',clim=(-3,1),width=.003)
        # Fixed subset; displacement is magnified by the same factor for every arm.
        idx=np.arange(0,10000,25)
        axes[1,col].quiver(before[idx,0],before[idx,1],movement[idx,0]*2,movement[idx,1]*2,angles='xy',scale_units='xy',scale=1,color='tab:blue',width=.003)
        np.savez(out/f'{method}.npz',grid=points,descent=vectors,before=before,after=after,mass=mass)
        summary.append(dict(method=method,weak_modes=int(weak.sum()),median_displacement=float(np.median(np.linalg.norm(movement,axis=1))),max_displacement=float(np.linalg.norm(movement,axis=1).max())))
    axes[0,0].set_ylabel('Coordinate descent direction (normalized arrows)')
    axes[1,0].set_ylabel('Actual G-only Adam step (movement ×2)')
    fig.suptitle('10×10 grid • 1.2M checkpoint • seed 0 fixed in advance\nGray: generated samples; red circles: accepted mass <0.08%; black crosses: mode centers')
    fig.tight_layout(rect=(0,0.06,1,.94))
    cb=fig.colorbar(q,cax=fig.add_axes([.35,.025,.30,.015]),orientation='horizontal');cb.set_label('log10 coordinate-gradient magnitude (color clipped to [−3, 1])')
    fig.savefig(out/'gradient_comparison.png',dpi=160);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'README.md').write_text('''# Frozen-discriminator gradient diagnostic

All four methods, seed 0 fixed in advance, at 1.2M updates. No training files modified.
Top: negative gradient of generator BCE with respect to fake coordinates on a 45×45 grid. Paired averages 128 common uniform real references and both slot assignments. Arrows normalized; color shows magnitude. This is a Monte Carlo diagnostic, not a reconstructed training trajectory.
Bottom: displacement of fixed latent samples after one G-only update on a copied checkpoint, with original Adam moments restored, batch 256 and D frozen. Common diagnostic random draws across arms. Arrows magnified 2×, same scale across methods. This omits the preceding D update of a normal training round. Adam momentum can disagree with instantaneous coordinate gradients.
Red circles: modes below 0.0008 accepted mass in the fixed 10k sample evaluation (not necessarily zero probability). Gray samples and marked modes are pre-update. Raw arrays retained in NPZ files. Seed 0 alone cannot establish a general explanation of missing modes.

![Diagnostic](gradient_comparison.png)
''')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
