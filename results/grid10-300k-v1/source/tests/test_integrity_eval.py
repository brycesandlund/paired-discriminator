import numpy as np
import torch
from paired_discriminator.integrity_eval import balanced_ids, nearest, discriminator_audit


def test_balanced_reference_order():
    labels=np.repeat(np.arange(10),12)
    ids=balanced_ids(labels,10)
    assert len(np.unique(ids))==100
    np.testing.assert_array_equal(labels[ids],np.arange(100)%10)


def test_nearest_matches_bruteforce_and_exact_copy():
    rng=np.random.default_rng(3)
    ref=rng.normal(size=(19,5)).astype('float32')
    query=np.concatenate([ref[[7]],rng.normal(size=(3,5)).astype('float32')])
    ids,dist=nearest(query,ref,device='cpu',batch=2)
    expected=np.linalg.norm(query[:,None]-ref[None,:],axis=2)
    np.testing.assert_array_equal(ids,expected.argmin(1))
    np.testing.assert_allclose(dist,expected.min(1),rtol=1e-6,atol=1e-7)
    assert ids[0]==7 and dist[0]==0


def test_disc_gap_and_slot_orientation():
    class Unary(torch.nn.Module):
        def forward(self,x,y): return x.mean((1,2,3))
    class Pair(torch.nn.Module):
        def forward(self,x,y): return x[:,:3].mean((1,2,3))-x[:,3:].mean((1,2,3))
    fake=torch.zeros(20,3,32,32)
    train=np.full((20,3,32,32),255,dtype='uint8')
    held=np.zeros_like(train)
    for method,d in [('vanilla',Unary()),('rsgan',Unary()),('paired',Pair())]:
        result,arrays=discriminator_audit(d,fake,train,held,method,'cpu')
        assert result['train_membership_auc_from_margin']==1
        assert result['train_minus_heldout_margin']==2
        assert result['train_real_acceptance']==1 and result['heldout_real_acceptance']==0
        equal,_=discriminator_audit(d,fake,train,train,method,'cpu')
        assert equal['train_membership_auc_from_margin']==.5
        assert equal['train_minus_heldout_margin']==0
