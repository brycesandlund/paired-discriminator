import json
import numpy as np
import pytest
import torch
from paired_discriminator import grid_d_batch as larger
from paired_discriminator import grid_pacgan, grid_dual_slot


def config():
    result=json.loads((larger.ROOT/'configs/grid3-50k.json').read_text())
    result.update(batch_size=16, steps=3, eval_every=1, eval_samples=32,
                  generator_width=16, discriminator_width=16)
    return result


@pytest.mark.parametrize('method',['paired','pacgan2','dual_slot'])
def test_batch256_setting_exactly_reproduces_original(tmp_path, method):
    c=config()
    old=grid_dual_slot if method=='dual_slot' else grid_pacgan
    old.run(c,method,0,tmp_path/'old')
    larger.run({**c,'discriminator_batch_size':16},method,0,tmp_path/'new')
    a=torch.load(tmp_path/'old/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'new/checkpoint.pt',weights_only=False)
    for component in ('generator','discriminator','rng_states'):
        for key in a[component]:
            assert torch.equal(a[component][key],b[component][key])
    assert np.array_equal(np.load(tmp_path/'old/final_samples.npy'),np.load(tmp_path/'new/final_samples.npy'))


@pytest.mark.parametrize('method',['paired','pacgan2','dual_slot'])
def test_only_d_batch_doubles_and_g_inputs_stay_identical(tmp_path,monkeypatch,method):
    c=config(); seen=[]
    old_d,old_g=larger.discriminator_loss,larger.generator_loss
    def d(discriminator,real,fake,method,slots):
        seen.append(('D',len(real),len(fake),len(slots)))
        return old_d(discriminator,real,fake,method,slots)
    def g(discriminator,real,fake,method,slots):
        seen.append(('G',len(real),len(fake),len(slots)))
        return old_g(discriminator,real,fake,method,slots)
    monkeypatch.setattr(larger,'discriminator_loss',d)
    monkeypatch.setattr(larger,'generator_loss',g)
    larger.run({**c,'discriminator_batch_size':32},method,0,tmp_path/'big')
    assert seen==[('D',32,32,32),('G',16,16,16)]*3
    larger.run(c,method,0,tmp_path/'base')
    a=torch.load(tmp_path/'big/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'base/checkpoint.pt',weights_only=False)
    for key in ('real_g','noise_g','slots','eval'):
        assert torch.equal(a['rng_states'][key],b['rng_states'][key])
    assert not torch.equal(a['rng_states']['noise_d'],b['rng_states']['noise_d'])
    for optimizer in ('optimizer_g','optimizer_d'):
        assert all(int(s['step'])==3 for s in a[optimizer]['state'].values())
