import numpy as np
from paired_discriminator.ring_mode_sweep_report import target_grid, spatial_tv, EDGES


def test_grid_accounts_for_outside_mass():
    config={'modes':8,'radius':2.,'sigma':.1}
    target=target_grid(config)
    assert np.isclose(target.sum(),1)
    assert target.min()>=0
    assert target[-1]<1e-10
    assert np.isclose(spatial_tv(np.full((100,2),10.),target),1)


def test_gaussian_grid_is_symmetric_and_matches_real_draws():
    target=target_grid({'modes':1,'radius':0.,'sigma':.1})
    grid=target[:-1].reshape(len(EDGES)-1,len(EDGES)-1)
    np.testing.assert_allclose(grid,grid[::-1],atol=1e-14)
    np.testing.assert_allclose(grid,grid.T,atol=1e-14)
    rng=np.random.default_rng(123)
    assert spatial_tv(rng.normal(0,.1,(100000,2)),target)<.025
    assert spatial_tv(rng.normal(0,.5,(100000,2)),target)>.6
