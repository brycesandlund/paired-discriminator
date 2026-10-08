import json
import numpy as np
import torch
from paired_discriminator.grid_reference import ROOT,centers
from paired_discriminator.grid10_report import EDGES,metrics,target_grid


def test_expanded_grid_density_and_coverage():
    c=json.loads((ROOT/'configs/grid10-150k.json').read_text())
    means=centers(c)
    assert means.shape==(100,2)
    assert float(means.min())==-6.75 and float(means.max())==6.75
    np.testing.assert_allclose(np.diff(EDGES),.05)
    target=target_grid(c,EDGES)
    assert target.sum()==1 and target[-1]<1e-12
    # Every outer mode remains inside the fine-density region.
    assert ((means.numpy()>EDGES[0]) & (means.numpy()<EDGES[-1])).all()
    points=means.repeat_interleave(100,dim=0).numpy()
    m,mass=metrics(points,c,target)
    assert m['mode_tv']<1e-12 and m['coverage_half_target']==100 and m['coverage_relative']==100
    # Half the mass invalid: half-target coverage remains complete, legacy 1% doesn't.
    points[::2]=100
    m,_=metrics(points,c,target)
    assert m['coverage_half_target']==100 and m['coverage_1pct']==0
    assert abs(m['mode_tv']-.5)<1e-12
