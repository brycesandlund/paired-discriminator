import pytest
import torch
from paired_discriminator import cifar_pair_arch as arch, cifar_tune

def test_matched_initialization_and_parameters():
    ds = {name: arch.models(name, 0)[1] for name in arch.ARCHITECTURES[2:]}
    base = ds['shared_difference']
    for d in ds.values():
        assert arch.state_equal(d.encoder.state_dict(), base.encoder.state_dict())
        assert arch.state_equal(d.head.state_dict(), base.head.state_dict())
    assert arch.state_equal(ds['self_attention'].state_dict(), ds['cross_attention'].state_dict())
    counts = [sum(p.numel() for p in ds[name].parameters()) for name in ('global_context','self_attention','cross_attention')]
    assert len(set(counts)) == 1

@pytest.mark.parametrize('name', arch.ARCHITECTURES)
def test_shapes_gradients_and_generator_identity(name):
    torch.set_num_threads(1)
    g, d = arch.models(name, 7)
    base, _ = arch.models('concat', 7)
    assert arch.state_equal(g.state_dict(), base.state_dict())
    x = torch.randn(3, 6, 32, 32, requires_grad=True)
    y = d(x); assert y.shape == (3,)
    grad = torch.autograd.grad(y.sum(), x)[0]
    assert torch.isfinite(grad).all()
    assert grad[:, :3].abs().sum() > 0 and grad[:, 3:].abs().sum() > 0
    if name not in ('concat', 'concat_wide'):
        assert torch.allclose(y, -d(torch.cat((x[:, 3:], x[:, :3]), 1)), atol=1e-6)

@pytest.mark.parametrize('name', ('shared_difference', 'self_attention', 'global_context', 'cross_attention'))
def test_true_reference_interaction(name):
    torch.set_num_threads(1)
    _, d = arch.models(name, 4)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    r = torch.randn_like(x)
    def grad(ref):
        return torch.autograd.grad(d(torch.cat((x, ref), 1)).sum(), x)[0]
    a, b = grad(r), grad(r.roll(1, 0))
    if name in ('shared_difference', 'self_attention'):
        assert torch.equal(a, b)
    else:
        assert (a - b).norm() > 1e-8

@pytest.mark.parametrize('name', ('concat', 'cross_attention'))
def test_exact_resume_and_control_equivalence(tmp_path, name):
    data = torch.randint(256, (8, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(123))
    a = arch.train(name, 0, 4, tmp_path/'full', data, device='cpu', batch=4)
    arch.train(name, 0, 2, tmp_path/'split', data, device='cpu', batch=4)
    b = arch.train(name, 0, 4, tmp_path/'split', data, device='cpu', batch=4)
    for k in ('model', 'discriminator', 'ema', 'rngs', 'optimizer_g', 'optimizer_d'):
        assert arch.state_equal(a[k], b[k])
    if name == 'concat':
        c = cifar_tune.train('paired', 0, cifar_tune.configs('paired')[0] | {'batch_size':4},
                            4, tmp_path/'old', data, device='cpu')
        for k in ('model', 'discriminator', 'ema', 'optimizer_g', 'optimizer_d'):
            assert arch.state_equal(a[k], c[k])
