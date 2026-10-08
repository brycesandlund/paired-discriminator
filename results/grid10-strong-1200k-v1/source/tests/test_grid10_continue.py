import json
import torch
import pytest
from paired_discriminator import grid_reference as original
from paired_discriminator.grid10_continue import run

@pytest.mark.parametrize('method',['vanilla','paired','vanilla_deficit','paired_deficit'])
def test_exact_continuation(tmp_path,method):
    c=json.loads((original.ROOT/'configs/grid10-150k.json').read_text())
    c.update(steps=6,eval_every=2,eval_samples=32,batch_size=16,generator_width=16,discriminator_width=16)
    original.run(c,method,0,tmp_path/'full')
    original.run({**c,'steps':3,'eval_every':1},method,0,tmp_path/'first')
    run(c,method,0,tmp_path/'continued',tmp_path/'first/checkpoint.pt')
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'continued/checkpoint.pt',weights_only=False)
    def same(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for key in ('generator','discriminator','optimizer_g','optimizer_d','rng_states','ema_mass'):same(a[key],b[key])
