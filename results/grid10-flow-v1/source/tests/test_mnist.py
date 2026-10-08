import json
from pathlib import Path
import numpy as np
import pytest
import torch
from paired_discriminator import mnist
from paired_discriminator.mnist_eval import distribution_metrics, DigitClassifier

@pytest.fixture
def config():
    c=json.loads((Path(__file__).parents[1]/'configs/mnist.json').read_text())
    return {**c,'width':4,'latent_dim':8,'batch_size':12,'log_every':1,'checkpoint_every':2,'eval_every':2}


def test_digit_metrics_known_cases():
    uniform=np.eye(10).repeat(100,axis=0)
    m=distribution_metrics(uniform,np.ones(10)/10)
    assert m['covered_digits']==10 and m['mode_tv']==0 and m['confidence_augmented_tv']==0
    one=np.eye(10)[np.zeros(100,dtype=int)]
    m=distribution_metrics(one,np.ones(10)/10)
    assert m['covered_digits']==1 and m['mode_tv']==pytest.approx(.9)
    uncertain=np.ones((100,10))/10
    m=distribution_metrics(uncertain,np.ones(10)/10)
    assert m['confidence_covered_digits']==0 and m['confidence_augmented_tv']==1
    target=np.arange(1,11)/55
    probs=np.repeat(np.eye(10),np.arange(1,11),axis=0)
    assert distribution_metrics(probs,target)['mode_tv']==0


def test_same_g_and_input_shapes(config):
    gs=[]
    for method in ('vanilla','paired'):
        g,d=mnist.models(config,method,0);g.eval();gs.append(g)
        x=g(torch.randn(2,8));assert x.shape==(2,1,28,28)
        assert d(torch.cat([x,x],1) if method=='paired' else x).shape==(2,)
    for k,v in gs[0].state_dict().items():torch.testing.assert_close(v,gs[1].state_dict()[k],rtol=0,atol=0)
    assert DigitClassifier()(torch.rand(2,1,28,28)).shape==(2,10)


@pytest.mark.parametrize('method',['vanilla','paired'])
def test_resume_and_sample_counts(config,tmp_path,method,monkeypatch):
    data=torch.randint(256,(20,1,28,28),dtype=torch.uint8,generator=torch.Generator().manual_seed(3))
    original=mnist.models;observed=[]
    def instrumented(*args):
        g,d=original(*args)
        g.register_forward_pre_hook(lambda m,a:observed.append(('g',m.training,torch.is_grad_enabled(),len(a[0]))))
        d.register_forward_pre_hook(lambda m,a:observed.append(('d',True,next(m.parameters()).requires_grad,len(a[0]))))
        return g,d
    monkeypatch.setattr(mnist,'models',instrumented)
    mnist.train(config,method,0,4,tmp_path/'full',data,device='cpu')
    assert [n for who,tr,grad,n in observed if who=='g' and tr and grad]==[12]*4
    assert [n for who,tr,grad,n in observed if who=='d' and grad]==[24 if method=='vanilla' else 12]*4
    mnist.train(config,method,0,2,tmp_path/'split',data,device='cpu')
    mnist.train(config,method,0,4,tmp_path/'split',data,device='cpu')
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);b=torch.load(tmp_path/'split/latest.pt',weights_only=False)
    for key in ('generator','discriminator','rngs'):
        for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert [(r['d_loss'],r['g_loss']) for r in a['rows']]==[(r['d_loss'],r['g_loss']) for r in b['rows']]


def test_losses_detach_d_and_pass_g_gradient(config):
    for method in ('vanilla','paired'):
        g,d=mnist.models(config,method,0);real=torch.randn(4,1,28,28);fake=g(torch.randn(4,8));slots=torch.tensor([0,1,0,1],dtype=torch.bool)
        mnist.loss_d(d,real,fake,method,slots).backward();assert all(p.grad is None for p in g.parameters())
        d.requires_grad_(False);mnist.loss_g(d,real,fake,method,slots).backward()
        assert sum(p.grad.abs().sum() for p in g.parameters())>0
