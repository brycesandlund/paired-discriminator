import json
import torch
from paired_discriminator import grid_reference_g as arm
from paired_discriminator import grid_reference as old


def test_only_g_reference_stream_changes(tmp_path,monkeypatch):
    c=json.loads((arm.ROOT/'configs/grid3-50k.json').read_text())
    c.update(steps=3,eval_every=1,eval_samples=32,batch_size=16,generator_width=16,discriminator_width=16,reference_phase='g_only',uniform_mix=.5,coverage_ema_decay=.99)
    seen=[]
    original=arm.discriminator_loss
    def track(d,real,fake,method,slots):
        seen.append(real.detach().clone())
        return original(d,real,fake,method,slots)
    monkeypatch.setattr(arm,'discriminator_loss',track)
    monkeypatch.setattr(old,'discriminator_loss',track)
    weights=[]
    weighted=arm.sample_weighted
    def sample(c,n,rng,w):
        weights.append(w.clone())
        return weighted(c,n,rng,w)
    monkeypatch.setattr(arm,'sample_weighted',sample)
    arm.run(c,'paired_deficit',0,tmp_path/'new')
    old.run(c,'paired',0,tmp_path/'base')
    assert len(weights)==3
    for a,b in zip(seen[:3],seen[3:]):assert torch.equal(a,b)
    a=torch.load(tmp_path/'new/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'base/checkpoint.pt',weights_only=False)
    for key in ('real_d','noise_d','noise_g','slots','eval'):
        assert torch.equal(a['rng_states'][key],b['rng_states'][key])
    assert not torch.equal(a['rng_states']['real_g'],b['rng_states']['real_g'])
    for key in ('optimizer_g','optimizer_d'):
        assert all(int(v['step'])==3 for v in a[key]['state'].values())
    assert torch.allclose(weights[0],torch.full((9,),1/9,dtype=torch.float64))
