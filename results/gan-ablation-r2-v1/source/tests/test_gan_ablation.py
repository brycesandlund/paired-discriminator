import json
import torch
from paired_discriminator.gan_ablation import make_models,train
from paired_discriminator.grid10_strong import run as original_run
from paired_discriminator.grid_pacgan import generator_loss
from paired_discriminator.grid_experiment import ROOT

def test_control_two_updates_match_original(tmp_path):
    c=json.loads((ROOT/'configs/grid10-strong-1200k.json').read_text())
    c.update(steps=2,eval_every=1,eval_samples=16,pilot_scale=4.3)
    (tmp_path/'config.json').write_text(json.dumps(c))
    train((str(tmp_path),dict(name='control'),0,2))
    original_run(c,'paired_deficit',0,tmp_path/'original')
    a=torch.load(tmp_path/'control_seed0/checkpoint_2.pt',weights_only=False)
    b=torch.load(tmp_path/'original/checkpoint.pt',weights_only=False)
    for key in ['generator','discriminator']:
        assert all(torch.equal(x,y) for x,y in zip(a[key].values(),b[key].values())) if key=='generator' else all(torch.equal(a[key]['net.'+k],v) for k,v in b[key].items())
    for key in a['rng']:assert torch.equal(a['rng'][key],b['rng_states'][key])
    torch.testing.assert_close(a['mass'],b['ema_mass'],rtol=0,atol=0)

def test_fourier_fake_gradient_exists_and_slots_work():
    c=json.loads((ROOT/'configs/grid10-strong-1200k.json').read_text());c['pilot_scale']=4.3
    g,d=make_models(c,dict(fourier=True),0)
    fake=g(torch.randn(8,16));fake.retain_grad()
    loss=generator_loss(d,torch.randn(8,2),fake,'paired',torch.arange(8)<4)
    loss.backward();assert torch.isfinite(fake.grad).all() and fake.grad.abs().sum()>0
