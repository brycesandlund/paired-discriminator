import json
import numpy as np
import torch
from paired_discriminator import experiment as ring
from paired_discriminator import grid_experiment as grid


def test_expansion_preserves_spacing_and_inner_modes():
    previous = set()
    for side in [3, 5, 7]:
        c = {'grid_side': side, 'modes': side * side, 'spacing': 1.5, 'sigma': .1,
             'valid_radius_sigma': 3, 'coverage_min_mass': .01}
        p = grid.centers(c)
        assert p.shape == (side * side, 2)
        assert set(map(tuple, p.tolist())) >= previous
        previous = set(map(tuple, p.tolist()))
        d = torch.cdist(p, p); d.fill_diagonal_(float('inf'))
        torch.testing.assert_close(d.min(1).values, torch.full((side*side,), 1.5))
        assert p.abs().max() == (side-1)*.75
        r = grid.coverage_metrics(p.repeat_interleave(100, dim=0), c)
        assert r['coverage'] == side * side
        assert abs(r['mode_tv']) < 1e-14


def test_grid_retains_original_models_and_losses():
    c = {'latent_dim':16, 'generator_width':128, 'discriminator_width':128}
    for method in ['vanilla', 'rsgan', 'paired']:
        g,d = grid.models(c, method, 3)
        oldg,oldd = ring.models(c, method, 3)
        for a,b in zip(g.parameters(), oldg.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(d.parameters(), oldd.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        real=torch.randn(10,2);fake=torch.randn(10,2);slots=torch.arange(10)%2
        for old,new in [(ring.discriminator_loss,grid.discriminator_loss),(ring.generator_loss,grid.generator_loss)]:
            torch.testing.assert_close(old(d,real,fake,method,slots),new(d,real,fake,method,slots),rtol=0,atol=0)
