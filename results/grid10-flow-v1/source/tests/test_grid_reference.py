import json
import numpy as np
import pytest
import torch
from paired_discriminator import grid_reference as ref
from paired_discriminator import grid_pacgan as baseline


def test_matching_bijection_cost_and_source_swap_equivariance():
    rng=torch.Generator().manual_seed(942)
    a=torch.randn(64,2,generator=rng);b=torch.randn(64,2,generator=rng)
    matched_a=ref.near_match(a,b)
    assert sorted(map(tuple,matched_a.tolist()))==sorted(map(tuple,a.tolist()))
    assert (matched_a-b).square().sum() <= (a-b).square().sum()
    # Reversing the sources gives exactly the same unordered matched edges.
    matched_b=ref.near_match(b,a)
    forward=sorted((tuple(x),tuple(y)) for x,y in zip(matched_a.tolist(),b.tolist()))
    backward=sorted((tuple(x),tuple(y)) for x,y in zip(a.tolist(),matched_b.tolist()))
    assert forward==backward


def test_deficits_floor_normalization_and_direction():
    weights=ref.deficit_weights(torch.tensor([.5,.5,0.,0.],dtype=torch.float64))
    torch.testing.assert_close(weights,torch.tensor([.125,.125,.375,.375],dtype=torch.float64))
    torch.testing.assert_close(ref.deficit_weights(torch.full((4,),.25)),torch.full((4,),.25))
    assert weights.sum()==1
    assert (weights>=.5/4).all()


def test_coverage_counts_invalid_mass_in_denominator():
    c=json.loads((ref.ROOT/'configs/grid3-50k.json').read_text())
    fake=torch.cat((ref.centers(c)[:2],torch.tensor([[100.,100.],[100.,100.]])))
    mass=ref.accepted_mass(fake,c)
    torch.testing.assert_close(mass[:2],torch.tensor([.25,.25],dtype=torch.float64))
    assert mass.sum()==.5


@pytest.mark.parametrize('method',['vanilla','paired'])
def test_random_policy_exactly_replays_baseline(tmp_path,method):
    c=json.loads((ref.ROOT/'configs/grid3-50k.json').read_text())
    c.update(batch_size=16,steps=3,eval_every=1,eval_samples=32,generator_width=16,discriminator_width=16)
    baseline.run(c,method,0,tmp_path/'old');ref.run(c,method,0,tmp_path/'new')
    a=torch.load(tmp_path/'old/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'new/checkpoint.pt',weights_only=False)
    for key in ('generator','discriminator','rng_states'):
        for k in a[key]:assert torch.equal(a[key][k],b[key][k])


@pytest.mark.parametrize('method',['paired_near','paired_deficit','vanilla_deficit'])
def test_d_only_keeps_g_sampling_streams_and_batch(tmp_path,method,monkeypatch):
    c=json.loads((ref.ROOT/'configs/grid3-50k.json').read_text())
    c.update(batch_size=16,steps=3,eval_every=1,eval_samples=32,generator_width=16,discriminator_width=16,reference_phase='d_only')
    seen=[];original=ref.generator_loss
    def track(d,real,fake,m,slots):
        seen.append((real.detach().clone(),len(fake),slots.detach().clone()))
        return original(d,real,fake,m,slots)
    monkeypatch.setattr(ref,'generator_loss',track)
    ref.run(c,method,0,tmp_path/'arm')
    ref.run(c,ref.base_method(method),0,tmp_path/'base')
    for a,b in zip(seen[:3],seen[3:]):
        assert torch.equal(a[0],b[0]) and a[1]==b[1]==16 and torch.equal(a[2],b[2])
    a=torch.load(tmp_path/'arm/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'base/checkpoint.pt',weights_only=False)
    for key in ('noise_d','noise_g','real_g','slots','eval'):
        assert torch.equal(a['rng_states'][key],b['rng_states'][key])
    if method.endswith('deficit'):assert a['ema_mass'].sum()<1
