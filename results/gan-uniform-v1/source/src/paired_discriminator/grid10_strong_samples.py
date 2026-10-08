"""Fixed seed-0 scatter comparison including both new deficit interventions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid_experiment import ROOT,centers

def main():
    out=ROOT/'results/grid10-strong-comparison-v1'
    config=json.loads((ROOT/'configs/grid10-strong-1200k.json').read_text())
    means=centers(config).numpy()
    panels=[('Vanilla','grid10-1200k-v1','vanilla'),('Paired','grid10-1200k-v1','paired'),('Vanilla + D deficit','grid10-1200k-v1','vanilla_deficit'),('Paired + D deficit (α=.5, p=1)','grid10-1200k-v1','paired_deficit'),('Paired + D and G deficit (α=.5, p=1)','grid10-both-1200k-v1','paired_deficit'),('Paired + stronger D deficit (α=.9, p=2)','grid10-strong-1200k-v1','paired_deficit')]
    fig,axes=plt.subplots(3,2,figsize=(12,17))
    for ax,(title,folder,method) in zip(axes.flat,panels):
        points=np.load(ROOT/f'results/{folder}/{method}_seed0/samples_step1200000.npy')
        assert points.shape==(10000,2) and np.isfinite(points).all()
        ax.scatter(points[:,0],points[:,1],s=2,alpha=.25,color='tab:blue',edgecolors='none')
        ax.scatter(means[:,0],means[:,1],s=9,color='red',edgecolors='none')
        ax.set(xlim=(-7.75,7.75),ylim=(-7.75,7.75),aspect='equal',title=title)
    fig.suptitle('1.2M updates • 10,000 fixed samples per method\nSeed 0, not selected for performance',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'samples_seed0_all_methods.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
