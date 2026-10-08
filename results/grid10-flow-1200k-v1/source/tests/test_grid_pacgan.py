import pytest
import torch
from torch import nn
from torch.nn import functional as F
from paired_discriminator import grid_pacgan as pac
from paired_discriminator import grid_experiment as base


def test_initialization_matches_paired():
    config = {'latent_dim': 16, 'generator_width': 128, 'discriminator_width': 128}
    for seed in range(5):
        for actual, expected in zip(pac.models(config, 'pacgan2', seed), base.models(config, 'paired', seed)):
            for a, b in zip(actual.parameters(), expected.parameters()):
                assert torch.equal(a, b)


def test_packing_preserves_every_point_once():
    points = torch.arange(512).reshape(256, 2)
    assert torch.equal(pac.pack(points), points.reshape(128, 4))
    with pytest.raises(ValueError):
        pac.pack(points[:255])


def test_discriminator_workload_loss_and_detachment():
    discriminator = nn.Linear(4, 1)
    real = torch.randn(256, 2)
    fake = torch.randn(256, 2, requires_grad=True)
    seen = []
    hook = discriminator.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
    actual = pac.discriminator_loss(discriminator, real, fake, 'pacgan2', None)
    hook.remove()
    assert seen == [torch.Size([256, 4])]
    expected = (F.softplus(-discriminator(real.reshape(128, 4))).mean()
                + F.softplus(discriminator(fake.reshape(128, 4))).mean()) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert fake.grad is None
    assert discriminator.weight.grad is not None


def test_generator_updates_both_pack_members():
    discriminator = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        discriminator.weight.copy_(torch.tensor([[1., 2., 3., 4.]]))
    discriminator.requires_grad_(False)
    fake = torch.zeros(256, 2, requires_grad=True)
    seen = []
    hook = discriminator.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
    loss = pac.generator_loss(discriminator, None, fake, 'pacgan2', None)
    hook.remove()
    assert seen == [torch.Size([128, 4])]
    loss.backward()
    expected = (-.5 / 128 * torch.tensor([1., 2., 3., 4.])).repeat(128).reshape(256, 2)
    torch.testing.assert_close(fake.grad, expected)
    assert discriminator.weight.grad is None
