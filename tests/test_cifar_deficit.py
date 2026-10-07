import json
from pathlib import Path
import pytest
import torch
from torch import nn
from paired_discriminator import cifar
from paired_discriminator.cifar_deficit import train, deficit_weights, accepted_mass, sample_indices, class_groups, classifier_probabilities

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,10)
        with torch.no_grad():
            self.linear.weight.fill_(.01)
            self.linear.bias.copy_(torch.arange(10).float()*3)
    def forward(self,x):
        return self.linear(x.mean((2,3)))


def config():
    c=json.loads((Path(__file__).parents[1]/'configs/cifar10-deficit.json').read_text())
    return {**c,'width':4,'latent_dim':8,'batch_size':4,'checkpoint_every':2,'eval_every':2,'log_every':1}


def test_deficit_mass_and_floor():
    p=torch.full((4,10),.01)
    p[0,0]=.91; p[1,0]=.91; p[2,1]=.91
    p[3]=.1
    mass=accepted_mass(p)
    torch.testing.assert_close(mass[:2],torch.tensor([.5,.25],dtype=torch.float64))
    assert mass.sum()==.75
    weights=deficit_weights(mass)
    assert weights.sum().item()==pytest.approx(1)
    assert weights.min().item()==pytest.approx(.05)
    assert weights[2]>weights[0]
    torch.testing.assert_close(deficit_weights(torch.full((10,),.1)),torch.full((10,),.1))
    labels=torch.arange(10).repeat_interleave(3)
    ids=sample_indices(class_groups(labels),torch.nn.functional.one_hot(torch.tensor(7),10).double(),100,torch.Generator().manual_seed(1))
    assert (labels[ids]==7).all()


def test_classifier_is_detached():
    fake=torch.randn(4,3,32,32,requires_grad=True)
    model=Classifier().eval()
    p=classifier_probabilities(model,fake)
    assert not p.requires_grad
    assert all(v.grad is None for v in model.parameters())


@pytest.mark.parametrize('method',['vanilla','paired','vanilla_deficit','paired_deficit'])
def test_resume_and_uniform_equivalence(tmp_path,method):
    c=config()
    data=torch.randint(256,(20,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(99))
    labels=torch.arange(10).repeat_interleave(2)
    classifier=Classifier()
    train(c,method,0,4,tmp_path/'full',data,labels,classifier,device='cpu')
    train(c,method,0,2,tmp_path/'split',data,labels,classifier,device='cpu')
    train(c,method,0,4,tmp_path/'split',data,labels,classifier,device='cpu')
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False)
    b=torch.load(tmp_path/'split/latest.pt',weights_only=False)
    torch.testing.assert_close(a['ema'],b['ema'],rtol=0,atol=0)
    for key in ('generator','discriminator','rngs'):
        for name in a[key]:
            torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    for ar,br in zip(a['rows'],b['rows']):
        assert ar['d_loss']==br['d_loss'] and ar['g_loss']==br['g_loss']
    assert a['ema'].max() > a['ema'].min()
    assert not classifier.training and all(not p.requires_grad for p in classifier.parameters())
    if not method.endswith('_deficit'):
        cifar.train(c,method,0,4,tmp_path/'old',data,device='cpu')
        old=torch.load(tmp_path/'old/latest.pt',weights_only=False)
        for key in ('generator','discriminator','rngs'):
            for name in a[key]:
                torch.testing.assert_close(a[key][name],old[key][name],rtol=0,atol=0)
