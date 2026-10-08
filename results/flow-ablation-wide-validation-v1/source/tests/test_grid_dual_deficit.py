import json
import torch
from paired_discriminator import grid_dual_deficit as arm
from paired_discriminator import grid_dual_slot as old


def test_d_only_preserves_g_and_slot_workload(tmp_path,monkeypatch):
    c=json.loads((arm.ROOT/'configs/grid3-50k.json').read_text())
    c.update(steps=3,eval_every=1,eval_samples=32,batch_size=16,generator_width=16,discriminator_width=16,reference_phase='d_only',uniform_mix=.5,coverage_ema_decay=.99)
    seen=[];original=arm.generator_loss
    def track(d,real,fake,method,slots):
        seen.append(real.detach().clone())
        return original(d,real,fake,method,slots)
    monkeypatch.setattr(arm,'generator_loss',track);monkeypatch.setattr(old,'generator_loss',track)
    batches=[];sample=arm.sample_weighted
    def weighted(c,n,rng,w):
        result=sample(c,n,rng,w);batches.append(result.clone());return result
    monkeypatch.setattr(arm,'sample_weighted',weighted)
    arm.run(c,'dual_slot',0,tmp_path/'new');old.run(c,'dual_slot',0,tmp_path/'old')
    assert len(batches)==3
    for a,b in zip(seen[:3],seen[3:]):assert torch.equal(a,b)
    for real in batches:
        fake=torch.arange(32).reshape(16,2).float()+100
        inputs,targets=arm.slot_batch(real,fake)
        assert inputs.shape==(16,4) and targets.shape==(16,2)
        assert torch.equal(inputs[:4].reshape(8,2),real[:8])
        assert torch.equal(inputs[8:12,2:],real[8:12])
        assert torch.equal(inputs[12:,:2],real[12:])
        assert targets.sum()==16
    a=torch.load(tmp_path/'new/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'old/checkpoint.pt',weights_only=False)
    for key in ('real_g','noise_d','noise_g','slots','eval'):
        assert torch.equal(a['rng_states'][key],b['rng_states'][key])
    assert not torch.equal(a['rng_states']['real_d'],b['rng_states']['real_d'])
    for key in ('optimizer_g','optimizer_d'):
        assert all(int(s['step'])==3 for s in a[key]['state'].values())
