import argparse,json,hashlib,time
import numpy as np
import torch
from .flow_ablation import Field
from .grid10_flow import sample
from .grid_experiment import ROOT
from .grid10_1200k_report import metrics,EDGES
from .grid_mode_sweep_report import target_grid

def build(name):
    torch.set_num_threads(1);out=ROOT/'results/flow-ablation-comparison-v1';rows=[]
    for seed in range(5):
        if name=='refine':root=ROOT/'results/flow-ablation-refine-v1';step=100000
        else:root=ROOT/'results'/('flow-ablation-r2-v1' if seed==0 else 'flow-ablation-wide-validation-v1');step=50000
        path=root/f'{name}_seed{seed}'
        hashes=json.loads((root/'provenance.json').read_text())['source_sha256']
        for f,h in hashes.items():assert hashlib.sha256((root/'source'/f).read_bytes()).hexdigest()==h
        ck=torch.load(path/f'checkpoint_{step}.pt',weights_only=False);assert ck['step']==step and ck['seed']==seed
        assert all(int(s['step'])==step for s in ck['optimizer']['state'].values())
        if name=='refine':
            resume=json.loads((path/'resume.json').read_text());assert resume['boundary_replay']
            from pathlib import Path
            assert hashlib.sha256(Path(resume['source']).read_bytes()).hexdigest()==resume['sha256']
        model=Field(ck['spec'],ck['scale']);model.load_state_dict(ck['model']);c=ck['config'];target=target_grid(c,EDGES)
        z=torch.randn(10000,2,generator=torch.Generator().manual_seed(70000+seed))
        assert np.array_equal(sample(model,z,128).numpy(),np.load(path/f'samples_{step}_raw.npy'))
        z=torch.randn(10000,2,generator=torch.Generator().manual_seed(902000+seed))
        for solver in (128,256):
            tick=time.perf_counter();points=sample(model,z,solver).numpy();secs=time.perf_counter()-tick;m,_=metrics(points,c,target)
            rows.append(dict(name=name,seed=seed,solver_steps=solver,step=step,sampling_seconds=secs,**m))
            np.save(out/f'{name}_seed{seed}_solver{solver}.npy',points)
        print(name,seed,rows[-2]['mode_tv'],rows[-2]['valid_fraction'],flush=True)
        (out/f'validation_metrics_{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/f'verification_{name}.json').write_text(json.dumps(dict(exact_replays=5,source_hashes=True,optimizer_steps=True,resume_boundaries=5 if name=='refine' else 0))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['refine','wide256']);build(p.parse_args().name)
