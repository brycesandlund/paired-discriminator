import json
import numpy as np
import torch
from paired_discriminator.grid_experiment import run as original
from paired_discriminator.grid_extension import run


def test_exact_continuation_all_methods(tmp_path):
    c=json.load(open('configs/grid7-50k.json'))
    c.update(steps=40,eval_every=10,eval_samples=100,batch_size=16,generator_width=16,discriminator_width=16)
    for method in ['vanilla','rsgan','paired']:
        full=tmp_path/method/'full'; first=tmp_path/method/'first'; split=tmp_path/method/'split'
        original(c,method,0,full)
        original({**c,'steps':20},method,0,first)
        run(c,method,0,split,resume_from=first)
        a=torch.load(full/'checkpoint.pt',weights_only=False);b=torch.load(split/'checkpoint.pt',weights_only=False)
        def compare(x,y):
            if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for k in x:compare(x[k],y[k])
            elif isinstance(x,(list,tuple)):
                assert len(x)==len(y)
                for u,v in zip(x,y):compare(u,v)
            else:assert x==y
        compare(a,b)
        np.testing.assert_array_equal(np.load(full/'final_samples.npy'),np.load(split/'final_samples.npy'))
