import torch
import pytest
from paired_discriminator.cifar_tune import train,configs
from paired_discriminator.cifar import train as original

@pytest.mark.parametrize('method',['vanilla','paired','flow'])
def test_resume_and_baseline_equivalence(tmp_path,method):
    torch.set_num_threads(1)
    data=torch.randint(256,(8,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(123))
    cfg=configs(method)[0]|{'batch_size':4}
    a=train(method,0,cfg,4,tmp_path/'full',data,device='cpu')
    train(method,0,cfg,2,tmp_path/'split',data,device='cpu')
    b=train(method,0,cfg,4,tmp_path/'split',data,device='cpu')
    for key in ['model','ema','rngs']:
        assert all(torch.equal(a[key][k],b[key][k]) for k in a[key])
    if method!='flow':
        oldcfg=dict(latent_dim=128,width=64,batch_size=4,learning_rate=2e-4,betas=[.5,.999],
                    log_every=4,checkpoint_every=4,eval_seed=99173)
        original(oldcfg,method,0,4,tmp_path/'original',data,device='cpu')
        old=torch.load(tmp_path/'original/latest.pt',weights_only=False)
        for ours,theirs in [('model','generator'),('discriminator','discriminator')]:
            assert all(torch.equal(a[ours][k],old[theirs][k]) for k in a[ours])

@pytest.mark.parametrize('method',['vanilla','paired'])
def test_r1_update_is_finite_and_changes_discriminator(tmp_path,method):
    data=torch.zeros(4,3,32,32,dtype=torch.uint8)
    c=configs(method)[-1]|{'batch_size':2}
    a=train(method,0,c,16,tmp_path/'r1',data,device='cpu')
    b=train(method,0,c|{'r1':0.},16,tmp_path/'no_r1',data,device='cpu')
    assert all(torch.isfinite(v).all() for v in a['model'].values())
    assert any(not torch.equal(a['discriminator'][k],b['discriminator'][k]) for k in a['discriminator'])
