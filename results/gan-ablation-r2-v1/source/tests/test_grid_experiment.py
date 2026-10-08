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


def test_only_geometry_changes_in_training_loop():
    import inspect
    for name in ['run', 'sample_real', 'models', 'mlp', 'pair_inputs', 'discriminator_loss', 'generator_loss', 'coverage_metrics']:
        assert inspect.getsource(getattr(grid, name)) == inspect.getsource(getattr(ring, name))


def test_spatial_metric_uses_the_expanding_grid():
    from paired_discriminator.grid_mode_sweep_report import target_grid, spatial_tv, EDGES
    config = {'grid_side':7, 'modes':49, 'spacing':1.5, 'sigma':.1}
    target = target_grid(config)
    assert np.isclose(target.sum(), 1)
    assert target[-1] < 1e-10
    assert target.min() >= 0
    matrix = target[:-1].reshape(len(EDGES)-1, len(EDGES)-1)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
    rng = np.random.default_rng(901)
    p = grid.centers(config).numpy()[rng.integers(49, size=100000)] + rng.normal(0, .1, (100000,2))
    assert spatial_tv(p, target) < .15
    assert spatial_tv(np.zeros((10000, 2)), target) > .99
