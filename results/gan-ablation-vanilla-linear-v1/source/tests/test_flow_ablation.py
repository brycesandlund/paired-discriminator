import torch
from paired_discriminator.flow_ablation import Field
from paired_discriminator.grid10_flow_oracle import Oracle
from paired_discriminator.grid10_flow import sample

def test_exact_field_gaussian_conditional():
    # One centered Gaussian has closed-form Gaussian interpolation dynamics.
    oracle=Oracle(dict(grid_side=1,modes=1,spacing=1.5,sigma=.3))
    x=torch.tensor([[1.,-2.],[.5,.4]]);t=torch.tensor([[.2],[.8]])
    variance=(1-t)**2+(t*.3)**2
    expected=(t*.09-(1-t))/variance*x
    torch.testing.assert_close(oracle(x,t),expected)
    z=torch.randn(20,2)
    torch.testing.assert_close(sample(oracle,z,256),z*.3,atol=5e-5,rtol=5e-5)

def test_features_and_preconditioning_gradients():
    for spec in [dict(depth=2),dict(depth=3,normalize=True,fourier=True),dict(depth=3,precondition=True,fourier=True)]:
        net=Field(spec,4.3);x=torch.randn(8,2,requires_grad=True);t=torch.linspace(0,1,8)[:,None]
        y=net(x,t);assert y.shape==x.shape and torch.isfinite(y).all()
        y.square().mean().backward();assert torch.isfinite(x.grad).all()

def test_gaussian_base_transports_scale_with_zero_residual():
    net=Field(dict(depth=2,precondition=True),4.3)
    for parameter in net.parameters():parameter.data.zero_()
    z=torch.randn(16,2)
    torch.testing.assert_close(sample(net,z,256),z*4.3,atol=5e-4,rtol=5e-4)
