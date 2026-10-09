import torch
from paired_discriminator import cifar_pair_arch as base, cifar_pair_symmetry as sym


def test_loop_is_frozen_and_control_is_identical(tmp_path):
    import inspect
    assert inspect.getsource(sym.train) == inspect.getsource(base.train)
    assert inspect.getsource(sym.diagnostics) == inspect.getsource(base.diagnostics)
    data=torch.randint(256,(8,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(123))
    a=base.train('concat',0,4,tmp_path/'base',data,device='cpu',batch=4)
    b=sym.train('concat',0,4,tmp_path/'sym',data,device='cpu',batch=4)
    for k in ('model','discriminator','ema','rngs','optimizer_g','optimizer_d'):
        assert base.state_equal(a[k],b[k])


def test_symmetry_gradient_and_resume(tmp_path):
    torch.set_num_threads(1)
    g,d=sym.models('symmetric_concat',0)
    oldg,oldd=base.models('concat',0)
    assert base.state_equal(g.state_dict(),oldg.state_dict())
    assert base.state_equal(d.core.state_dict(),oldd.state_dict())
    x=torch.randn(4,6,32,32,requires_grad=True)
    swapped=torch.cat((x[:,3:],x[:,:3]),1)
    assert torch.allclose(d(x),-d(swapped),atol=1e-6)
    grad=torch.autograd.grad(d(x).sum(),x)[0]
    assert torch.isfinite(grad).all() and grad[:,:3].norm()>0 and grad[:,3:].norm()>0
    data=torch.randint(256,(8,3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(123))
    a=sym.train('symmetric_concat',0,4,tmp_path/'full',data,device='cpu',batch=4)
    sym.train('symmetric_concat',0,2,tmp_path/'split',data,device='cpu',batch=4)
    b=sym.train('symmetric_concat',0,4,tmp_path/'split',data,device='cpu',batch=4)
    for k in ('model','discriminator','ema','rngs','optimizer_g','optimizer_d'):
        assert base.state_equal(a[k],b[k])
