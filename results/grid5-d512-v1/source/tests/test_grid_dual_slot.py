import pytest
import torch
from torch import nn
from torch.nn import functional as F
from paired_discriminator import grid_dual_slot as dual
from paired_discriminator import grid_pacgan as pac


def test_same_generator_and_discriminator_hidden_initialization():
    config = {'latent_dim': 16, 'generator_width': 128, 'discriminator_width': 128}
    for seed in range(5):
        g, d = dual.models(config, 'dual_slot', seed)
        pg, pd = pac.models(config, 'paired', seed)
        for a, b in zip(g.parameters(), pg.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(d[:4].parameters(), pd[:4].parameters()):
            assert torch.equal(a, b)
        assert d[-1].out_features == 2
        assert sum(p.numel() for p in d.parameters()) == 17410


@pytest.mark.parametrize('generator_phase', [False, True])
def test_pair_sources_targets_and_exact_sample_accounting(generator_phase):
    real = torch.arange(512).reshape(256, 2).float() + 1000
    fake = torch.arange(512).reshape(256, 2).float() - 1000
    inputs, labels = dual.slot_batch(real, fake, generator_phase=generator_phase)
    points = inputs.reshape(-1, 2)
    flat_labels = labels.flatten()
    assert inputs.shape == (192 if generator_phase else 256, 4)
    assert torch.equal(flat_labels.bool(), points[:, 0] >= 1000)
    actual_fake = points[flat_labels == 0]
    assert torch.equal(actual_fake.sort(dim=0).values, fake.sort(dim=0).values)
    expected_real = real[:128] if generator_phase else real
    assert torch.equal(points[flat_labels == 1].sort(dim=0).values, expected_real.sort(dim=0).values)
    combinations, counts = labels.unique(dim=0, return_counts=True)
    assert len(combinations) == (3 if generator_phase else 4)
    assert (counts == 64).all()


def test_discriminator_loss_and_detachment():
    d = nn.Linear(4, 2)
    real, fake = torch.randn(256, 2), torch.randn(256, 2, requires_grad=True)
    inputs, labels = dual.slot_batch(real, fake)
    expected = F.binary_cross_entropy_with_logits(d(inputs), labels, reduction='sum') / 512
    actual = dual.discriminator_loss(d, real, fake, 'dual_slot', None)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert fake.grad is None
    assert d.weight.grad is not None


def test_generator_mask_and_cross_slot_gradients():
    d = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        d.weight.copy_(torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
    d.requires_grad_(False)
    fake = torch.zeros(256, 2, requires_grad=True)
    real = torch.zeros_like(fake)
    loss = dual.generator_loss(d, real, fake, 'dual_slot', None)
    loss.backward()
    # FF: both heads act on both inputs. FR: only A head. RF: only B head.
    expected = torch.cat((torch.tensor([6., 8., 10., 12.]).repeat(64).reshape(128, 2),
                          torch.tensor([1., 2.]).repeat(64, 1),
                          torch.tensor([7., 8.]).repeat(64, 1))) * (-.5 / 256)
    torch.testing.assert_close(fake.grad, expected)
    assert d.weight.grad is None


def test_invalid_batches_rejected():
    with pytest.raises(ValueError):
        dual.slot_batch(torch.zeros(6, 2), torch.zeros(6, 2))
