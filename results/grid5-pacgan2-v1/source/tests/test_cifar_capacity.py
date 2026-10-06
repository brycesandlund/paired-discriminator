import json
from pathlib import Path
import pytest
import torch
from paired_discriminator import cifar_capacity as current
from paired_discriminator import cifar_integrity as previous

BASE = json.loads((Path(__file__).parents[1] / 'configs/cifar10-projection.json').read_text())

@pytest.mark.parametrize('d_batch,d_width', [(12,4),(24,4),(12,8)])
def test_batches_resume_and_baseline_replay(tmp_path, monkeypatch, d_batch, d_width):
    config = {**BASE, 'width':4, 'latent_dim':8, 'batch_size':12,
              'discriminator_batch_size':d_batch, 'discriminator_width':d_width,
              'log_every':1, 'checkpoint_every':2}
    labels = torch.arange(10).repeat(2)
    data = torch.randint(256,(20,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(9))
    original = current.models
    forwards = []
    d_forwards = []
    def instrumented(*args):
        g,d = original(*args)
        g.register_forward_pre_hook(lambda m,a: forwards.append((m.training,torch.is_grad_enabled(),len(a[0]))))
        d.register_forward_pre_hook(lambda m,a: d_forwards.append((next(m.parameters()).requires_grad,len(a[0]))))
        return g,d
    monkeypatch.setattr(current,'models',instrumented)
    current.train(config,'paired',0,4,tmp_path/'full',data,labels,device='cpu')
    assert [b for training,grad,b in forwards if training and grad] == [12]*4
    assert [b for training,grad,b in forwards if training and not grad] == [12]*(4*d_batch//12)
    assert [b for grad,b in d_forwards if grad] == [d_batch]*4
    assert [b for grad,b in d_forwards if not grad] == [12]*4
    current.train(config,'paired',0,2,tmp_path/'split',data,labels,device='cpu')
    current.train(config,'paired',0,4,tmp_path/'split',data,labels,device='cpu')
    a = torch.load(tmp_path/'full/latest.pt',weights_only=False)
    b = torch.load(tmp_path/'split/latest.pt',weights_only=False)
    def equal_states(a,b):
        for k in ('generator','discriminator','rngs'):
            for name in a[k]: torch.testing.assert_close(a[k][name],b[k][name],rtol=0,atol=0)
        assert [(r['d_loss'],r['g_loss']) for r in a['rows']] == [(r['d_loss'],r['g_loss']) for r in b['rows']]
    equal_states(a,b)
    if d_batch==12 and d_width==4:
        previous.train(config,'paired',0,4,tmp_path/'previous',data,labels,device='cpu')
        equal_states(a,torch.load(tmp_path/'previous/latest.pt',weights_only=False))

@pytest.mark.parametrize('arm', ['d_batch256','d_width128'])
def test_real_arm_architecture_and_initialization(arm):
    config=json.loads((Path(__file__).parents[1]/f'configs/cifar10-{arm}.json').read_text())
    g,d=current.models(config,'paired',0)
    oldg,oldd=previous.models(BASE,'paired',0)
    for k,v in g.state_dict().items(): torch.testing.assert_close(v,oldg.state_dict()[k],rtol=0,atol=0)
    assert config['batch_size']==128 and config['width']==64
    assert sum(p.numel() for p in g.parameters())==1191683
    if arm=='d_batch256':
        for k,v in d.state_dict().items(): torch.testing.assert_close(v,oldd.state_dict()[k],rtol=0,atol=0)
    else:
        assert d.features[0].out_channels==128 and d.embedding.embedding_dim==512
