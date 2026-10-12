import pytest
import torch
from paired_discriminator import cifar_budget as budget, cifar_tune as old


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def check(a,b):
    for k in ('model','ema','discriminator','optimizer_g','optimizer_d','rngs'):
        assert equal(a[k],b[k]),k


@pytest.mark.parametrize('method',['paired','flow'])
def test_parity_and_resume(tmp_path,method):
    data=torch.randint(256,(8,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(42))
    cfg=budget.config(method)|{'batch_size':4}
    full=budget.train(method,3,cfg,4,tmp_path/'full',data,device='cpu')
    budget.train(method,3,cfg,2,tmp_path/'split',data,device='cpu')
    split=budget.train(method,3,cfg,4,tmp_path/'split',data,device='cpu')
    baseline=old.train(method,3,cfg,4,tmp_path/'old',data,device='cpu')
    check(full,split);check(full,baseline)


def test_reset_matches_previous_recipe_and_resumes(tmp_path):
    data=torch.zeros(8,3,32,32,dtype=torch.uint8)
    cfg=budget.config('paired')|{'batch_size':4,'reset_step':2}
    full=budget.train('paired',3,cfg,4,tmp_path/'full',data,device='cpu')
    budget.train('paired',3,cfg,3,tmp_path/'split',data,device='cpu')
    split=budget.train('paired',3,cfg,4,tmp_path/'split',data,device='cpu')
    old.train('paired',3,cfg,2,tmp_path/'base',data,device='cpu')
    branch=old.train('paired',3,cfg,2,tmp_path/'branch',data,parent=tmp_path/'base/latest.pt',device='cpu')
    check(full,split);check(full,branch)


def test_budget_resume_does_not_spend_again(tmp_path):
    data=torch.zeros(8,3,32,32,dtype=torch.uint8)
    cfg=budget.config('paired')|{'batch_size':4}
    a=budget.train('paired',3,cfg,1000,tmp_path/'run',data,device='cpu',budget_seconds=.001)
    assert a['step']==10 and a['training_seconds']>=.001
    b=budget.train('paired',3,cfg,1000,tmp_path/'run',data,device='cpu',budget_seconds=.001)
    assert a['step']==b['step'] and a['training_seconds']==b['training_seconds']
    check(a,b)
